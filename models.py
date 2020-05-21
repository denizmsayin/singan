import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import exact_interpolate


class Conv2DBlock(nn.Module):
    """ Combine Conv2d-BN-LReLU into a single block """

    # the 0.2 negative slope is given in the supplementary materials
    def __init__(self, in_channels, out_channels, kernel_size,  # conv arguments
                 use_bn=False, activation=None,  # customization of following blocks
                 conv_kwargs=None, bn_kwargs=None):  # optional kwargs for conv and bn

        # mutable default arguments are dangerous
        if conv_kwargs is None:
            conv_kwargs = {}
        if bn_kwargs is None:
            bn_kwargs = {}

        # call superclass init and (maybe) create layers
        super().__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)
        self.bn = nn.BatchNorm2d(out_channels, **bn_kwargs) if use_bn else nn.Identity()
        self.activ = activation if activation else nn.Identity()

    def forward(self, x):
        return self.activ(self.bn(self.conv(x)))


class SGNet(nn.Module):
    """
    A class to create the networks used in the SinGAN paper. Each generator and 
    discriminator is very similar, being composed of 5 blocks of 
    (conv2d, batch_norm, leaky_relu) blocks, with the final one being slightly different.
    All intermediate blocks have the same amount of kernels and a kernel size of 3x3.
    Zero padding is done initially, so that the network preserves the shape of its input.
    """

    def __init__(self, num_blocks=5, kernel_count=32, kernel_size=3, # architecture customization
                 final_activation=nn.Tanh(), final_bn=False,  # final layer cust.
                 input_channels=3, output_channels=3):  # channel counts

        # superclass init and add the initial padding layer
        super().__init__()
        layers = [nn.ZeroPad2d(num_blocks)]  # since kernel size is 3, pad 1 per block

        # loop to create each layer except last, 
        # all properties are shared except for the number of channels
        def sgnet_block(in_channels, out_channels):
            return Conv2DBlock(in_channels, out_channels, kernel_size,
                               activation=nn.LeakyReLU(negative_slope=0.2)) # as given in the paper

        layers.append(sgnet_block(input_channels, kernel_count))  # first layer
        for _ in range(num_blocks - 2):  # last layer has a different architecture
            layers.append(sgnet_block(kernel_count, kernel_count))
        # the final activation depends on whether this is the generator or critic
        # (tanh for gen. and none for crit.), and is different from the others
        final_block = Conv2DBlock(kernel_count, output_channels, kernel_size,
                                  final_bn, final_activation)
        layers.append(final_block)

        # create a sequential model from it
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # simply forwards through the layers
        return self.model(x)


def noise_sampler(noise_std):
    """
    This function provides a common interface from which we draw the noise samplers,
    to make it easy to control all the sampling from one code block, we could easily
    change from normal to uniform just by changing one line here, for example.
    A noise sampler simply takes a reference tensor and produces noise with the same shape.
    """
    def sample_like(x):
        return noise_std * torch.randn_like(x)
    return sample_like


class SGGen(torch.nn.Module):
    """
    This class adds the extra fluff (noise sampling and residual connections)
    on top of the basic SGNet architecture to create the full single-scale generator.
    """
    def __init__(self, sgnet, noise_std):
        super().__init__()
        self.sgnet = sgnet
        self.noise_sampler = noise_sampler(noise_std)

    def forward(self, x, z=None):
        if z is None:
            z = self.noise_sampler(x)
        g_in = x + z  # image + noise as input
        g_out = self.sgnet(g_in) + x  # residual connection
        return g_out


class MultiScaleSGGenView(nn.Module):
    """ 
    This class serves as a 'view' over the list of generators that makes the stack
    look like a single generator. Multiple scales of generators are combined by
    starting from the lowest; the output of the lower scale is resized and 
    passed to the upper generator automatically until no more are left. 
    In the end we have something that takes an image input and returns
    another image, just like a single generator. 
    
    Attributes:
        generators: a list of SGGen's representing generator networks, converted
            to nn.ModuleList when stored
        scaling_factor: a floating point scalar which represents the scale multiplier
            between each generator (e.g. 1.25)
        scaling_mode: a string for the scaling mode, should be a valid input for
            torch.nn.functional.interpolate's 

    Illustration of the full architecture:
            samplerN -> noiseN -> | generatorN | 
                        imgN-1 -> |            | -> imgN
                         ^
                         .............................
                         .......other generators......
                         .............................
                                                     ^
            sampler1 -> noise1 -> | generator1 |     |
                        img0   -> |            | -> img1
                         ^
                         |____________________________
                                                     ^
            sampler0 -> noise0 -> | generator0 |     |
                        img0   -> |            | -> img0

    Note about scaling:
        
        Simply using scaling_factor to scale outputs is nice when we do not
        have any strict requirements on image shapes, but does not really
        work when we expect a certain size for each output. Consider
        the starting from a size of 250 and scaling by a factor of 3/4:
        
        scales = [250, 188, 141, 105, 79, 59, 44, 33, 25]

        Since we round the result at each step, the final output is 25, although
        250 * 0.75^8 ~ 25.08. If we take an input with size 25 and scale up with
        a factor 4/3 we get the following:

        scales = [25, 33, 44, 59, 79, 105, 140, 187, 250]

        Notice that some scales do not match because we started with 25 instead of
        25.08. This can be a problem when calculating reconstruction loss, for
        example. Thus, we provide an optional argument to the forward pass, a
        (float, float) tuple for providing the exact size (e.g. (25.08, 25.08) 
        rather than (25, 25) to be used when upsampling) to ensure that we obtain
        exact shape matches.

    """

    def __init__(self, generators, scaling_factor, scaling_mode='bicubic'):

        # initialize superclass and check arguments
        super().__init__()

        # assign members, nn.ModuleList for generators to ensure
        # proper behavior, e.g. .parameters() returning correctly
        self.generators = nn.ModuleList(generators)
        self.scaling_factor = scaling_factor
        self.scaling_mode = scaling_mode

        # freeze all generators except for the top one 
        for g in self.generators[:-1]:
            g.requires_grad_(False)
            g.eval()

    def forward(self, x, exact_size=None, z_input=None):
        """
        Forward pass through the network.

        Args: 
        x: a 4D (N, C, H, W) tensor input to the first (coarsest scale) generator,
        z_input: a list of 4D noise tensors to be used as the noise input at each scale,
            if None, the noise samplers are used to generate noise
        exact_size: a (float, float) tuple for providing the theoretical shape of the input,
            see the 'Note about scaling:' in the class docstring.
            if None, the size of x is used as the exact_size
        """
        # set exact_size as the input size if not provided
        if exact_size is None:
            exact_size = tuple(float(d) for d in x.shape[2:4])  # (H, W)

        # go through each generator
        x_out = None
        for i, g, in enumerate(self.generators):
            z = None if z_input is None else z_input[i]  # get the noise input from the proper source
            x_out = g(x, z)  # pass through
            if i < len(self.generators) - 1:  # upsample if not the last layer
                # interpolate using the exact dimensions and update them
                x, exact_size = exact_interpolate(x_out, self.scaling_factor, exact_size, self.scaling_mode)
        return x_out


class FixedSizeSGGenView(nn.Module):
    """
    A wrapper to fix the size of an SGNet view for easier calls to forward, so that
    we do not have to provide the coarsest zero input and exact size at each call
    """
    def __init__(self, sgnet_view, coarsest_example_input, coarsest_exact_size):
        super().__init__()
        self.sgnet_view = sgnet_view
        self.coarsest_exact_size = coarsest_exact_size
        self.coarsest_zero_input = torch.zeros_like(coarsest_example_input)

    def forward(self, z_input=None):
        return self.sgnet_view.forward(self.coarsest_zero_input, self.coarsest_exact_size, z_input)
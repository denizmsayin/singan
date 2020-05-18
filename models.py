import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import exact_interpolate


class Conv2DBlock(nn.Module):
    """ Combine Conv2d-BN-LReLU into a single block """

    # the 0.2 negative slope is given in the supplementary materials
    def __init__(self, in_channels, out_channels, kernel_size,  # conv arguments
                 use_bn=True, activation=None,  # customization of following blocks
                 conv_initializer=None, bn_initializer=None,  # possibly custom inits
                 conv_kwargs=None, bn_kwargs=None):  # kwargs for conv and bn

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

        # call custom initializers if they exist
        if conv_initializer:
            conv_initializer(self.conv)
        if use_bn and bn_initializer:
            bn_initializer(self.bn)

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

    def __init__(self, num_blocks=5, kernel_count=32,  # architecture customization
                 final_activation=nn.Tanh(), final_bn=False,  # final layer cust.
                 input_channels=3, output_channels=3,  # channel counts
                 conv_init=None, bn_init=None):  # custom inits

        # superclass init and add the initial padding layer
        super().__init__()
        layers = [nn.ZeroPad2d(num_blocks)]  # since kernel size is 3, pad 1 per block

        # loop to create each layer except last, 
        # all properties are shared except for the number of channels
        def sgnet_block(in_channels, out_channels):
            return Conv2DBlock(in_channels, out_channels, 3,
                               activation=nn.LeakyReLU(negative_slope=0.2),  # as given in the paper
                               conv_initializer=conv_init, bn_initializer=bn_init)

        layers.append(sgnet_block(input_channels, kernel_count))  # first layer
        for _ in range(num_blocks - 2):  # last layer has a different architecture
            layers.append(sgnet_block(kernel_count, kernel_count))
        # the final activation depends on whether this is the generator or critic
        # (tanh for gen. and none for crit.), and is different from the others
        final_block = Conv2DBlock(kernel_count, output_channels, 3,
                                  final_bn, final_activation,
                                  conv_init, bn_init)
        layers.append(final_block)

        # create a sequential model from it
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # simply forwards through the layers
        return self.model(x)


class MultiScaleSGNetView(nn.Module):
    """ 
    This class serves as a 'view' over the list of generators that makes the stack
    look like a single generator. Multiple scales of generators are combined by
    starting from the lowest; the output of the lower scale is resized and 
    passed to the upper generator automatically until no more are left. 
    In the end we have something that takes an image input and returns
    another image, just like a single generator. 
    
    Attributes:
        generators: a list of nn.Module's representing generator networks, converted
            to nn.ModuleList when stored
        scaling_factor: a floating point scalar which represents the scale multiplier
            between each generator (e.g. 1.25)
        noise_samplers: a list of functions that take a tensor as input and return 
            noise of the same shape as the input, one for each generator. 
            The order of samplers should match with the order of the generators.
        scaling_mode: a string for the scaling mode, should be a valid input for
            torch.nn.functional.interpolate's 

    Illustration of the view:
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

    def __init__(self, generators, scaling_factor, noise_samplers, scaling_mode='bicubic'):

        # initialize superclass and check arguments
        super().__init__()
        assert len(generators) == len(noise_samplers), \
            'Number of generators and noise samplers do not match'

        # assign members, nn.ModuleList for generators to ensure
        # proper behavior, e.g. .parameters() returning correctly
        self.generators = nn.ModuleList(generators)
        self.scaling_factor = scaling_factor
        self.noise_samplers = noise_samplers
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

            # get the noise input from the proper source
            if z_input is None:
                z = self.noise_samplers[i](x)
            else:
                z = z_input[i]

            # pass through and upsample for the next scale
            g_input = x + z  # add the noise and input image
            x_out = g(g_input) + x  # add the gen. output and the input image

            if i < len(self.generators) - 1:  # upsample if not the last layer
                # interpolate using the exact dimensions and update them
                x, exact_size = exact_interpolate(x_out, exact_size,
                                                  self.scaling_factor,
                                                  self.scaling_mode)

        return x_out

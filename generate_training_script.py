import re
import json
from sys import argv, exit

# if the first line of a code cell corresponds to this,
# then it is considered a training cell
ARG_CELL_MARKER = '# arguments cell'
TRAIN_CELL_MARKER = '# training cell'
TRAIN_FILE_NAME = 'train.py'
INDENTATION = 2


def is_marked_cell(cell, marker):
    return cell['cell_type'] == 'code' and cell['source'] and cell['source'][0].strip() == marker


def get_marked_cells(notebook, marker):
    return [cell for cell in notebook['cells'] if is_marked_cell(cell, marker)]


def is_comment(line):
    return line.strip().startswith('#')


def extract_args(arg_cells):
    args = []
    for cell in arg_cells:
        for line in cell['source']:
            if is_comment(line):
                continue
            if line.count('=') == 1:  # assignment found
                clean_line = strip_trailing_comment(line)
                pair = tuple([s.strip() for s in clean_line.split('=')])
                args.append(pair)
    return args


def strip_trailing_comment(line):
    # this is difficult to do reliably because a # can be nested in a string
    # and there are escape characters and stuff, a compiler backend is needed
    # but I'll keep it simple and unreliable instead!
    pos = line.find('#')
    if pos != -1:
        return line[:pos].strip()
    return line.strip()


_pattern_lookups = [
    ('int', re.compile(r'^[-+]?\d+$')), # int
    ('float', re.compile(r'^[-+]?\d+.\d*|\d+/\d+$')), # rational or float
    ('str', re.compile(r'^\'.*\'|".*"]$')), # string
]


def detect_type(str_value):
    for typ, pattern in _pattern_lookups:
        if re.match(pattern, str_value):
            return typ
    raise ValueError(f'Could not match type of: {str_value}')


def create_parser_arg(parser, arg_name, arg_value):
    arg_type = detect_type(arg_value)
    return f'{parser}.add_argument(\'--{arg_name}\', default={arg_value}, type={arg_type})'


def indent(s):
    return INDENTATION * ' ' + s


class CodeBuilder:
    def __init__(self):
        self.level = 0
        self.code = []

    def indent(self):
        self.level += 1

    def unindent(self):
        self.level = max(self.level-1, 0)

    def add(self, s):
        fmt_s = self.level * INDENTATION * ' ' + s + '\n'
        self.code.append(fmt_s)

    def build(self):
        return ''.join(self.code)

def make_parser_code(args):
    parser = 'parser'
    builder = CodeBuilder()
    builder.add('def parse_arguments():')
    builder.indent()
    builder.add(f'{parser} = ArgumentParser()')
    for name, value in args:
        builder.add(create_parser_arg(parser, name, value))
    builder.add(f'return {parser}')
    return builder.build()


def make_arg_pattern(args):
    arg_pattern_strings = [r'\W' + name + r'\W' for name, _ in args]
    combined_pattern = re.compile(r'|'.join(arg_pattern_strings))
    return combined_pattern

def make_train_code(args, training_cells):
    training_code = '\n'.join([''.join(cell['source']) for cell in training_cells])
    arg_pattern = make_arg_pattern(args)
    return training_code

def make_main_code():
    builder = CodeBuilder()
    builder.add('if __name__ == \'__main__\':')
    builder.indent()
    builder.add('args = parse_arguments()')



if __name__ == '__main__':
    if len(argv) != 2:
        print('usage: python generate_training_script.py notebook_path')
        exit()

    _, notebook_path = argv
    with open(notebook_path, 'r') as notebook_file:
        notebook = json.load(notebook_file)

    arg_cells = get_marked_cells(notebook, ARG_CELL_MARKER)
    args = extract_args(arg_cells)
    lower_args = [(n.lower(), v) for n, v in args]
    parser_code = make_parser_code(lower_args)

    training_cells = get_marked_cells(notebook, TRAIN_CELL_MARKER)

    with open(TRAIN_FILE_NAME, 'w') as train_file:
        train_file.write('def training(opts):')
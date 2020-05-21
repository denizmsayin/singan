import re
import json
from sys import argv, exit

# if the first line of a code cell corresponds to this,
# then it is considered a training cell
ARG_CELL_PATTERN = re.compile(r'^# arguments cell$')
TRAIN_CELL_PATTERN = re.compile(r'^# training cell$')
IMPORT_PATTERN = re.compile(r'^import .*$')
TRAIN_FILE_NAME = 'train.py'
INDENTATION = 2


def is_marked_cell(cell, marker):
    return cell['cell_type'] == 'code' and cell['source'] and re.match(marker, cell['source'][0].strip())


def get_code_cells_with_matching_first_line(notebook, marker):
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
    ('int', re.compile(r'^[-+]?\d+$')),  # int
    ('float', re.compile(r'^[-+]?\d+.\d*|\d+/\d+$')),  # rational or float
    ('str', re.compile(r'^\'.*\'|".*"]$')),  # string
]


def detect_type(str_value):
    for typ, pattern in _pattern_lookups:
        if re.match(pattern, str_value):
            return typ
    raise ValueError(f'Could not match type of: {str_value}')


def create_parser_arg(parser, arg_name, arg_value):
    arg_type = detect_type(arg_value)
    return f'{parser}.add_argument(\'--{arg_name}\', default={arg_value}, type={arg_type})'


class CodeBuilder:
    def __init__(self):
        self.level = 0
        self.code = []

    def indent(self):
        self.level += 1

    def unindent(self):
        self.level = max(self.level-1, 0)

    def add(self, s):
        if not s.endswith('\n'):
            s += '\n'
        fmt_s = self.level * INDENTATION * ' ' + s
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
    builder.add(f'return {parser}.parse_args()')
    return builder.build()


def make_arg_pattern(args):
    or_pattern_string = r'|'.join([name for name, _ in args])
    combined_pattern = re.compile(r'(\W)(' + or_pattern_string + r')(\W)')
    return combined_pattern


_remove_pattern = re.compile(r'^\s*(?:plt.*)?(?:#.*)?\s*$')


def substitute_arg(m):
    return m.group(1) + 'args.' + m.group(2).lower() + m.group(3)


def make_train_code(args, training_cells):
    # extract and combine code cells
    notebook_code = []
    for cell in training_cells:
        for line in cell['source']:
            # filter the code, removing empties, comments and plt stuff
            if re.match(_remove_pattern, line):
                continue
            notebook_code.append(line)

    # substitute ARG with args.ARG
    arg_pattern = make_arg_pattern(args)
    for i in range(len(notebook_code)):
        notebook_code[i] = re.sub(arg_pattern, substitute_arg, notebook_code[i])

    # build the code
    builder = CodeBuilder()
    builder.add('def train(args):')
    builder.indent()
    for line in notebook_code:
        builder.add(line)
    return builder.build()


def make_main_code():
    builder = CodeBuilder()
    builder.add('if __name__ == \'__main__\':')
    builder.indent()
    builder.add('args = parse_arguments()')
    builder.add('train(args)')
    return builder.build()


def make_import_code(import_cells):
    builder = CodeBuilder()
    builder.add('from argparse import ArgumentParser')
    for cell in import_cells:
        for line in cell['source']:
            builder.add(line)
    return builder.build()


def convert_notebook(path):
    with open(path, 'r') as notebook_file:
        notebook = json.load(notebook_file)

    code_parts = []

    # imports
    import_cells = get_code_cells_with_matching_first_line(notebook, IMPORT_PATTERN)
    code_parts.append(make_import_code(import_cells))

    # parser
    arg_cells = get_code_cells_with_matching_first_line(notebook, ARG_CELL_PATTERN)
    args = extract_args(arg_cells)
    lower_args = [(n.lower(), v) for n, v in args]
    code_parts.append(make_parser_code(lower_args))

    # training code
    training_cells = get_code_cells_with_matching_first_line(notebook, TRAIN_CELL_PATTERN)
    code_parts.append(make_train_code(args, training_cells))

    # main_code
    code_parts.append(make_main_code())

    with open(TRAIN_FILE_NAME, 'w') as train_file:
        for code_part in code_parts:
            train_file.write(code_part)
            train_file.write('\n\n')


if __name__ == '__main__':
    if len(argv) != 2:
        print('usage: python generate_training_script.py notebook_path')
        exit()

    _, notebook_path = argv
    convert_notebook(notebook_path)

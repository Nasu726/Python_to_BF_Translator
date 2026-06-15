import sys

REALLOC_SIZE = 300000

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read()
    except IOError:
        print("File open error.", file=sys.stderr)
        sys.exit(1)

def brainfuck_interpreter(code):
    mem = [0] * 300000
    ptr = 0
    pc = 0
    code_len = len(code)

    while pc < code_len:
        cmd = code[pc]

        if cmd == '>':
            ptr += 1
        elif cmd == '<':
            ptr -= 1
        elif cmd == '+':
            mem[ptr] = (mem[ptr] + 1) % 256
        elif cmd == '-':
            mem[ptr] = (mem[ptr] - 1) % 256
        elif cmd == '.':
            print(chr(mem[ptr]), end='')
        elif cmd == ',':
            try:
                inp = input()[0]
                mem[ptr] = ord(inp)
            except IndexError:
                mem[ptr] = 0
        elif cmd == '[':
            if mem[ptr] == 0:
                rc = 1
                while rc != 0:
                    pc += 1
                    if pc >= code_len:
                        print("'[' is too many.", file=sys.stderr)
                        sys.exit(1)
                    if code[pc] == '[':
                        rc += 1
                    elif code[pc] == ']':
                        rc -= 1
        elif cmd == ']':
            if mem[ptr] != 0:
                rc = 1
                while rc != 0:
                    pc -= 1
                    if pc < 0:
                        print("']' is too many.", file=sys.stderr)
                        sys.exit(1)
                    if code[pc] == ']':
                        rc += 1
                    elif code[pc] == '[':
                        rc -= 1
                continue  # skip pc++ to re-evaluate
        pc += 1

def main():
    code = read_file("test.bf")
    brainfuck_interpreter(code)

if __name__ == "__main__":
    main()

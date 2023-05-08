import string

char_list = []
chars = string.printable[-38:-6]
for i,ch in enumerate(chars):
    char_list.append(ch)

print(char_list)

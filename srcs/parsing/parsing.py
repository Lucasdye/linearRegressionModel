
def is_int_or_float(str):
    """
    Summary: Checks if the string is a float or an int.
    If true, returns True and the concerned type found.
    Otherwie returns False.
    """
    if str == "":
        return False
    if str[0] == '+' or str[0] == '-':
        str = str[1:]
    dot = False
    for i in range(0, len(str)):
        if str[i].isdigit() is False:
            if str[i] == '.' and dot is True:
                return False
            if str[i] == '.' and dot is False:
                dot = True
            else:
                return False
    if dot == True:
        return {"nb": str, "type": "float"}
    else:
        return {"nb": str, "type": "int"}
		
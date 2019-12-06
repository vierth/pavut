def rename(names, input_div, output_div):
    return [name.replace(inputdiv,output_div) for name in names]
    
def c_rename(names):
    newnames = []
    for name in names:
        name = name.split("_")
        newname = [name[0]]
        newname.extend(name[1].split("-"))
        
        newnames.append("_".join(newname))
    return newnames
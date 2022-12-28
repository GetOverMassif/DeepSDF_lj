import os

if __name__ == "__main__":
    data_path = "/media/lj/TOSHIBA/dataset/ShapeNet/ShapeNetCore.v2/04256520"
    dirs = os.listdir(data_path)
    dirs.sort()
    with open('data.json', mode = 'w') as f:
        f.write('{\n')
        f.write('  "ShapeNetV2": {\n')
        f.write('    "04256520": [\n')
        for dir in dirs:
            str = '      "' + dir + '",' + '\n'
            f.write(str)
        f.write('    ]\n')
        f.write('  }\n')
        f.write('}\n')

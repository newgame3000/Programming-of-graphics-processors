#!/usr/bin/env python3

import click
from PIL import Image
from pathlib import Path, PosixPath


@click.command()
@click.argument('images', nargs=-1, type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
def convert(images):
    """\
        Конвертирует изображения во входной файл для программы. На изображении должны быть прозрачные пиксели 
        (прямоугольники). Каждый пиксель интерпретируется как класс (из прямоугольников берется левый верхний пиксель).

        В код зашито, что имя изображения оканчивается на _test, а выходной файл теста - то же имя, 
        оканчивающееся на _clusters

        Пиксели можно нарисовать, например, при помощи GIMP (https://stackoverflow.com/a/8097548).
        При экспорте необходимо убедится, что у прозрачных пикселей сохраняется цвет.
    """
    for image_file in images:
        classes = []
        with Image.open(image_file) as img:
            width, height = img.size
            pix = img.load()
            for i in range(width):
                for j in range(height):
                    if pix[i, j][3] == 0:
                        classes.append((i, j))
                        ax, ay = i, j
                        bx, by = 0, 0
                        for k in range(i+1, width):
                            if pix[k, j][3] != 0:
                                bx = k-1
                                break
                        else:
                            bx = width-1

                        for k in range(j+1, height):
                            if pix[i, k][3] != 0:
                                by = k-1
                                break
                        else:
                            by = height-1

                        for x in range(ax, bx+1):
                            for y in range(ay, by+1):
                                pix[x, y] = (*pix[x, y][:3], 255)

        with open(image_file.with_suffix('.in'), 'wt') as test:
            test.write(f"{PosixPath(image_file.name).with_suffix('.data')}\n")
            test.write(f"{str(PosixPath(image_file).name).replace('_test.png', '_clusters.data', 1)}\n")
            test.write(f'{len(classes)}\n')
            test.write('\n'.join([' '.join(map(str, i)) for i in classes]) + '\n')

        print(f'{image_file}: {len(classes)} classes')


if __name__ == '__main__':
    convert()

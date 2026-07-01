import argparse
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_voc_xml(xml_path, voc_root, datasets_root):
    """Parse a VOC XML annotation file and return a list of object dicts"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    abs_image_path = os.path.join(voc_root, 'JPEGImages', filename)
    image_path = os.path.relpath(abs_image_path, datasets_root)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = int(obj.find('difficult').text)
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        
        objects.append({
            'image_path': image_path,
            'file_name': filename,
            'width': width,
            'height': height,
            'class': name,
            'x_min': xmin,
            'y_min': ymin,
            'x_max': xmax,
            'y_max': ymax,
            'difficult': difficult
        })
    
    return objects


def parse_annotations_dir(voc_root, split_name, datasets_root):
    """
    Recorre todos los XMLs en VOC_ROOT/Annotations
    y construye image_path relativo a datasets_root
    """
    annotations_dir = os.path.join(voc_root, 'Annotations')
    all_objects = []
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    print(f"\nEncontrados {len(xml_files)} archivos XML en {annotations_dir}")
    
    for xml_file in tqdm(xml_files, desc=f"Procesando {split_name}"):
        xml_path = os.path.join(annotations_dir, xml_file)
        objects = parse_voc_xml(xml_path, voc_root, datasets_root)
        all_objects.extend(objects)
    
    return pd.DataFrame(all_objects, columns=[
        'image_path', 'file_name', 'width', 'height',
        'class', 'x_min', 'y_min', 'x_max', 'y_max', 'difficult'
    ])


def get_dominant_class(group):
    """Devuelve la clase más frecuente de una imagen"""
    return group['class'].value_counts().index[0]


def stratified_split_by_image(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split estratificado por imagen usando la clase dominante."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    image_classes = (
        df.groupby('file_name')
        .apply(get_dominant_class)
        .reset_index()
    )
    image_classes.columns = ['file_name', 'dominant_class']

    train_images, valtest_images = train_test_split(
        image_classes,
        test_size=(val_ratio + test_ratio),
        stratify=image_classes['dominant_class'],
        random_state=random_state
    )

    val_images, test_images = train_test_split(
        valtest_images,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=valtest_images['dominant_class'],
        random_state=random_state
    )

    train_df = df[df['file_name'].isin(train_images['file_name'])].reset_index(drop=True)
    val_df   = df[df['file_name'].isin(val_images['file_name'])].reset_index(drop=True)
    test_df  = df[df['file_name'].isin(test_images['file_name'])].reset_index(drop=True)

    return train_df, val_df, test_df


def print_stats(df, train_df, val_df, test_df, threshold=0.02):
    """Imprime tabla comparativa de distribución de clases entre splits"""
    
    classes = sorted(df['class'].unique())
    
    overall_dist = df['class'].value_counts(normalize=True)
    train_dist   = train_df['class'].value_counts(normalize=True)
    val_dist     = val_df['class'].value_counts(normalize=True)
    test_dist    = test_df['class'].value_counts(normalize=True)

    print(f"\n{'='*60}")
    print(f"{'Split':<10} {'Imágenes':>10} {'Objetos':>10}")
    print(f"{'-'*60}")
    print(f"{'Total':<10} {df['file_name'].nunique():>10} {len(df):>10}")
    print(f"{'Train':<10} {train_df['file_name'].nunique():>10} {len(train_df):>10}")
    print(f"{'Val':<10} {val_df['file_name'].nunique():>10} {len(val_df):>10}")
    print(f"{'Test':<10} {test_df['file_name'].nunique():>10} {len(test_df):>10}")
    print(f"{'='*60}")

    col_w = 12
    print(f"\nDistribución de clases por split (threshold ±{threshold*100:.0f}%):")
    print(f"\n{'Clase':<15} {'Overall':>{col_w}} {'Train':>{col_w}} {'Val':>{col_w}} {'Test':>{col_w}}")
    print(f"{'-'*60}")

    for cls in classes:
        overall = overall_dist.get(cls, 0)
        train   = train_dist.get(cls, 0)
        val     = val_dist.get(cls, 0)
        test    = test_dist.get(cls, 0)

        train_ok = abs(train - overall) <= threshold
        val_ok   = abs(val   - overall) <= threshold
        test_ok  = abs(test  - overall) <= threshold

        train_sym = '[ok]' if train_ok else '[x] '
        val_sym   = '[ok]' if val_ok   else '[x] '
        test_sym  = '[ok]' if test_ok  else '[x] '

        print(
            f"{cls:<15} "
            f"{overall:>{col_w}.3f} "
            f"{train_sym}{train:>{col_w-4}.3f} "
            f"{val_sym}{val:>{col_w-4}.3f} "
            f"{test_sym}{test:>{col_w-4}.3f}"
        )

    all_ok = all(
        abs(train_dist.get(cls, 0) - overall_dist.get(cls, 0)) <= threshold and
        abs(val_dist.get(cls, 0)   - overall_dist.get(cls, 0)) <= threshold and
        abs(test_dist.get(cls, 0)  - overall_dist.get(cls, 0)) <= threshold
        for cls in classes
    )
    print(f"\n{'='*60}")
    if all_ok:
        print(f"[ok] Todos los splits dentro del threshold ±{threshold*100:.0f}%")
    else:
        print(f"[x]  Algunos splits fuera del threshold ±{threshold*100:.0f}%")
    print(f"{'='*60}\n")


def prepare_datasets(trainval_root, test_root, outdir, datasets_root):
    """
    Parsea trainval + test, los unifica, hace split 80/10/10 y guarda 3 CSVs
    """
    os.makedirs(outdir, exist_ok=True)

    df_trainval = parse_annotations_dir(trainval_root, split_name='trainval', datasets_root=datasets_root)
    df_test     = parse_annotations_dir(test_root,     split_name='test',     datasets_root=datasets_root)

    df = pd.concat([df_trainval, df_test], ignore_index=True)
    print(f"\nTotal tras unificar:")
    print(f"  Objetos:          {len(df)}")
    print(f"  Imágenes únicas:  {df['file_name'].nunique()}")
    print(f"  Clases:           {sorted(df['class'].unique())}")

    print("\nHaciendo split estratificado...")
    train_df, val_df, test_df = stratified_split_by_image(df)

    train_df.to_csv(os.path.join(outdir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(outdir,   'val.csv'),   index=False)
    test_df.to_csv(os.path.join(outdir,  'test.csv'),  index=False)

    print_stats(df, train_df, val_df, test_df, threshold=0.02)

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_root', type=str, required=True,
                        help='Raíz base de los datasets (ej: ../DataSets)')
    parser.add_argument('--trainval_root', type=str, required=True,
                        help='Raíz de VOC trainval (ej: ../DataSets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012)')
    parser.add_argument('--test_root', type=str, required=True,
                        help='Raíz de VOC test (ej: ../DataSets/VOCtest_06-Nov-2007/VOCdevkit/VOC2007)')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Directorio donde guardar los 3 CSVs')

    args = parser.parse_args()

    prepare_datasets(
        args.trainval_root,
        args.test_root,
        args.outdir,
        args.datasets_root
    )


# Usage:
"""
python scripts/PrepareDataset.py \
    --datasets_root ../DataSets \
    --trainval_root ../DataSets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012 \
    --test_root     ../DataSets/VOCtest_06-Nov-2007/VOCdevkit/VOC2007 \
    --outdir        data/ProcessedCSVs
"""
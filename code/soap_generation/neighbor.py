import numpy as np
import os
from tqdm import tqdm


def num_common(l1, l2):
    n1 = 0
    for x in l1:
        if x in l2:
            n1 += 1

    n2 = 0
    for x in l2:
        if x in l1:
            n2 += 1
    if n1 == 3 and n2 == 3:
        return True
    else:
        return False


def get_neighbors(csv_location):
    filenames = []
    barriers = []
    for lineidx, line in enumerate(open(csv_location + "vaskas_features_properties_smiles_filenames.csv", "r")):
        if lineidx > 0:
            filenames.append(line.split()[0]+".xyz")
            barriers.append(float(line.split(",")[-3]))
    names = []
    lens = []
    smiles = []

    # Find ligands for each molecule
    side_groups = []
    for filename in filenames:
        l = filename.replace("ir_tbp_1_dft-", "").split()[0].split("_smi1")[
            0].replace("_1_dft", "").replace("_1", "").split("-")
        side_groups_here = []
        for x in l:
            if "_" in x:
                for y in x.split("_"):
                    if y not in names:
                        names.append(y)
                    side_groups_here.append(y)
            else:
                if x not in names:
                    names.append(x)
                side_groups_here.append(x)
        # if len(side_groups_here)!=4:
        #    print(filename)

        side_groups.append(side_groups_here)

    diffs = []
    diffs2 = []
    pairs = []
    bs = []
    for idx1, s1 in enumerate(side_groups):
        b1 = barriers[idx1]
        for idx2, s2 in enumerate(side_groups):
            if idx2 <= idx1:
                continue
            b2 = barriers[idx2]
            if num_common(s1, s2):
                # print(s1,s2)
                diff = b2-b1
                diffs.append(abs(diff))
                pairs.append([filenames[idx1], filenames[idx2]])
                diffs2.append([s1, s2])
                bs.append([b1, b2])

    for element in pairs:
        pair0 = element[0]
        pair1 = element[0]

    return np.array(diffs), pairs


def interpolate(element1, element2, alpha=0.5):
    return (element1 * alpha) + (element2 * (1 - alpha))


def get_interpolations(csv_location, elements, labels, names, interpolation_steps=5):
    diffs, pairs = get_neighbors(csv_location)

    names = np.array(names)
    labels = np.array(labels)

    elements_inter = []
    labels_inter = []
    for pair in tqdm(pairs):
        indices1 = np.where(names == pair[0].split(',')[-1])
        indices2 = np.where(names == pair[1].split(',')[-1])

        for index1 in indices1:
            for index2 in indices2:
                for x in np.linspace(0, 1, interpolation_steps):
                    elements_inter.append(interpolate(
                        elements[index1], elements[index2], alpha=x))
                    labels_inter.append(interpolate(
                        labels[index1], labels[index2], alpha=x))

    elements_inter = np.array(elements_inter[:30])
    labels_inter = np.array(labels_inter[:30])

    return elements_inter, labels_inter

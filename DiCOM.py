from pathlib import Path

import matplotlib.pyplot as plt
import pydicom


def show_image(dicom_file):
    ct = dicom_file.pixel_array
    plt.figure()
    # Alternatively cmap="bone"
    plt.imshow(ct, cmap="gray")
    # plt.imshow(ct, cmap="bone")
    plt.show()


def handle_3D_dicom():
    path_to_head_mri = Path("./data/DICOM/SE000001/")
    all_files = list(path_to_head_mri.glob("*"))
    print(all_files)
    print(len(all_files))

    mri_data = []
    for path in all_files:
        data = pydicom.read_file(path)
        mri_data.append(data)

    # the slice locations are unordered and useless, so entire scan is shuffled and useless
    print("Unordered slice location:")
    for slice_loc in mri_data[:5]:
        print(slice_loc.SliceLocation)

        # output
        # 89.9999955528687
        # 95.9999960937442
        # 29.9999952815023
        # 131.999997780749
        # 23.9999946081714

    # to make the slice locations ordered, we use sorted function of Python
    print("Ordered slice location:")
    mri_data_ordered = sorted(mri_data, key=lambda slice_loc: slice_loc.SliceLocation)
    for slice_loc in mri_data_ordered[:5]:
        print(slice_loc.SliceLocation)

        # output
        # 0.0
        # 5.99999663091323
        # 11.9999973042441
        # 17.9999979772582
        # 23.9999946081714

    # extract the actual data
    full_volume = []
    for slices in mri_data_ordered:
        full_volume.append(slices.pixel_array)

    # visualize
    fig, axis = plt.subplots(3, 3, figsize=(10, 10))
    slice_counter = 0
    for i in range(3):
        for j in range(3):
            axis[i][j].imshow(full_volume[slice_counter], cmap="gray")
            slice_counter += 1

    plt.show()


# Read one dicom file
# Handle 2D images
dicom_file = pydicom.read_file("./data/DICOM/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm")
print("Header:")
print(dicom_file)
print("--------------")
# convert to Hex to see an entry in dicom
print("Rows using unique identifier:")
print(dicom_file[0x0028, 0x0010])

print("Rows Entry:")
print(dicom_file.Rows)

# extract the image from dicom
# show_image(dicom_file)

# 3D image real life
# Handle 3D images
handle_3D_dicom()


import os
import torch

from models.dccnn import DCCNN
from models.compact_cnn import CompactCNNV7
from visualize_util import save_density_map_normalize, save_density_map
from torchvision import datasets, transforms
from PIL import Image
from torchvision.io import write_jpeg, read_image
from torch import nn

def overlay_img_with_density(img_origin, density_map_path, output_path):
    """

    :param img_path:
    :param density_map_path: output .torch of density map
    :param output_path:
    :return:
    """

    # # img_origin = Image.open(img_path).convert('RGB')
    #
    # transformer = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    img_tensor = read_image(img_origin)
    print("img tensor shape", img_tensor.shape)
    density_map_tensor = torch.load(density_map_path)

    print(img_tensor.shape)
    print(density_map_tensor.shape)
    print(density_map_tensor.sum())
    density_map_tensor = torch.from_numpy(density_map_tensor).unsqueeze(dim=0).unsqueeze(dim=0)
    print("density_map_tensor.shape", density_map_tensor.shape)  # torch.Size([1, 1, 46, 82])
    upsampling_density_map_tensor = nn.functional.interpolate(density_map_tensor, scale_factor=8) / 64

    overlay_density_map = img_tensor.detach().clone()
    upsampling_density_map_tensor = (upsampling_density_map_tensor.squeeze(dim=0) / upsampling_density_map_tensor.max() * 255)
    overlay_density_map[0] = torch.clamp_max(img_tensor[0] + upsampling_density_map_tensor[0] * 2, max=255)

    write_jpeg(overlay_density_map.type(torch.uint8), output_path, quality=100)

def overlay_img_with_density_padding(img_origin, density_map_path, output_path):
    """

    :param img_path:
    :param density_map_path: output .torch of density map
    :param output_path:
    :return:
    """

    # # img_origin = Image.open(img_path).convert('RGB')
    #
    # transformer = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    img_tensor = read_image(img_origin)
    print("img tensor shape", img_tensor.shape)
    density_map_tensor = torch.load(density_map_path)

    print(img_tensor.shape)
    print(density_map_tensor.shape)
    print(density_map_tensor.sum())
    density_map_tensor = torch.from_numpy(density_map_tensor).unsqueeze(dim=0).unsqueeze(dim=0)
    print("density_map_tensor.shape", density_map_tensor.shape)  # torch.Size([1, 1, 46, 82])
    upsampling_density_map_tensor = nn.functional.interpolate(density_map_tensor, scale_factor=8) / 64

    pad_density_map_tensor = torch.zeros((1, 3, img_tensor.shape[1], img_tensor.shape[2]))
    pad_density_map_tensor[:, 0, :upsampling_density_map_tensor.shape[2], :upsampling_density_map_tensor.shape[3]] = upsampling_density_map_tensor

    overlay_density_map = img_tensor.detach().clone()
    pad_density_map_tensor = (pad_density_map_tensor.squeeze(dim=0) / pad_density_map_tensor.max() * 255)

    overlay_density_map[0] = torch.clamp_max(img_tensor[0] + pad_density_map_tensor[0] * 2, max=255)

    write_jpeg(overlay_density_map.type(torch.uint8), output_path, quality=100)


def preprocess_input(image_path):
    img_origin = Image.open(image_path).convert('RGB')
    transformer = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])
    img_processed = transformer(img_origin).unsqueeze(0)
    return img_processed


def count_people(INPUT_NAME, OUTPUT_NAME, MODEL):
    NAME="adamw1_ccnnv7_t4_bike_prediction"
    # INPUT_NAME = "/data/my_crowd_image/dataset_batch1245/mybikedata/test_data/images/IMG_20201127_160829_821.jpg"
    # OUTPUT_NAME = "/data/my_crowd_image/dataset_batch1245/mybikedata/test_data/tmp/IMG_20201127_160829_821"
    # MODEL = "/data/save_model/adamw1_ccnnv7_t4_bike/adamw1_ccnnv7_t4_bike_checkpoint_valid_mae=-3.143752908706665.pth"

    loaded_file = torch.load(MODEL)
    # model = DCCNN()
    model = CompactCNNV7()
    model.load_state_dict(loaded_file['model'])
    model.eval()

    img = preprocess_input(INPUT_NAME)
    predict_path = OUTPUT_NAME
    pred = model(img)
    pred = pred.detach().numpy()[0][0]
    pred_count = pred.sum()
    print(pred_count)
    save_density_map(pred, predict_path)
    torch.save(pred, predict_path+".torch")
    overlay_img_with_density_padding(INPUT_NAME, predict_path+".torch", predict_path+".overlay.jpg")

    print("save to ", predict_path)
    return pred_count


if __name__ == "__main__":
    """
    predict all in folder 
    output into another folder 
    output density map and count in csv
    """
    NAME="adamw1_ccnnv7_t4_bike_prediction"
    INPUT_NAME = "/data/my_crowd_image/dataset_batch1245/mybikedata/test_data/images/IMG_20201127_160829_821.jpg"
    OUTPUT_NAME = "/data/my_crowd_image/dataset_batch1245/mybikedata/test_data/tmp/IMG_20201127_160829_821"
    MODEL = "/data/save_model/adamw1_ccnnv7_t4_bike/adamw1_ccnnv7_t4_bike_checkpoint_valid_mae=-3.143752908706665.pth"

    loaded_file = torch.load(MODEL)
    model = DCCNN()
    model.load_state_dict(loaded_file['model'])
    model.eval()

    img = preprocess_input(INPUT_NAME)
    predict_path = OUTPUT_NAME
    pred = model(img)
    pred = pred.detach().numpy()[0][0]
    pred_count = pred.sum()
    print(pred_count)
    save_density_map(pred, predict_path)
    torch.save(pred, predict_path+".torch")
    overlay_img_with_density(INPUT_NAME, predict_path+".torch", predict_path+".overlay.jpg")

    print("save to ", predict_path)




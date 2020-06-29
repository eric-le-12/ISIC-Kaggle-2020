import torch.nn as nn
import torch
import os
from sklearn.metrics import classification_report
from data_loader import dataloader
import pandas as pd
import json
from data_loader import transform
from model import classification as cls
from barbar import Bar

# use this file if you want to quickly test your model

with open("cfgs/tenes.cfg") as f:
    cfg = json.load(f)


def test_result(model, test_loader, device):
    # testing the model by turning model "Eval" mode
    model.eval()
    preds = []
    names = []
    for data, name in Bar(test_loader):
        # move-tensors-to-GPU
        data = data.to(device)
        output = model(data)
        output = torch.sigmoid(output)
        print(output)
        output = output > 0.5
        names.extend(list(name))
        preds.extend(output.tolist())
    
    test_result = pd.DataFrame({'image_name':names,'target':preds})
    test_result.to_csv('testing.csv')
    return (test_result)


def main():
    print("Testing process beginning here....")


if __name__ == "__main__":
    main()
    test_data = cfg["data"]["test_csv_name"]
    data_path = cfg["data"]["data_path"]
    test_df = pd.read_csv(test_data, usecols=["image_name", "target"])
    test_df['image_name'] = test_df['image_name']  +'.jpg'

    # prepare the dataset
    testing_set = dataloader.TestDataset(
        test_df, './dataset/test/test', transform.test_transform
    )
    # make dataloader
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=16, shuffle=False,)
    # load model
    extractor_name = cfg["train"]["extractor"]
    model = cls.ClassificationModel(model_name=extractor_name).create_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("saved/models", "20200628-1724-model1.pth")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # print classification report
    test_result(model, test_loader, device)

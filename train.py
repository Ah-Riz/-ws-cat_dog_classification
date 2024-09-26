import torch
import torch.nn as nn
from data_prep import data_prep
from tqdm.auto import tqdm
from timeit import default_timer as timer
from architecture import *
import yaml
from utils import convert_params

from dotenv import load_dotenv
import neptune
from neptune.utils import stringify_unsupported
import os

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()
    
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               device: torch.device):
    model.eval()
    
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            y_pred_logits = model(X)
            
            loss = loss_fn(y_pred_logits, y)
            test_loss += loss.item()
            
            test_pred_labels = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)
            
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int = 10,
          device: str = "cpu"):
    best_accuracy = 0
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
        
        model_name = "best_model.pt"
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), model_name)
        
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
    
    print(f"Best test accuracy: {best_accuracy:.4f}")
    
    return results, model_name

def main():
    load_dotenv()
    
    project_name = os.getenv("NEPTUNE_PROJECT")
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    
    run = neptune.init_run(project=project_name, api_token=api_token)
    
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    model_arch = {
        "MobileNetV2": MobileNetV2,
        "VGG16": VGG16
    }
    
    train_loader, val_loader = data_prep("./data/PetImages")
    torch.manual_seed(42)
    # print(config["arch"][config["choose_arch"]]["params"]["input_shape"])
    # print(config["arch"][[[config["choose_arch"]]]])
    choosen_arch = config["choose_arch"]
    method = [x for x in config["arch"] if x["name"] == choosen_arch][0]
    arch_name = method["name"]
    model_arch_version = config["model_version"]
    model_config = convert_params(method["params"])
    # model = model_arch[config["choose_arch"]]()
    model_namespace = f"models/{arch_name}/{model_arch_version}"
    
    modelClass = model_arch[arch_name]
    model = modelClass(**model_config)
    print(model)
    
    for param_name, param_value in model_config.items():
        run[f"{model_namespace}/params/{param_name}"] = param_value
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = 10
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start_time = timer()
    results, model_name = train(model=model,
                   train_dataloader=train_loader,
                   test_dataloader=val_loader,
                   optimizer=optimizer,
                   loss_fn=loss_fn,
                   epochs=NUM_EPOCHS,
                   device=device)
    
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    
    run[f"{model_namespace}/artifacts"].upload(model_name)
    run[f"{model_namespace}/classification_report"] = stringify_unsupported(results)
    run.stop()
    print(f"Classification report: {results}")

if __name__ == "__main__":
    main()
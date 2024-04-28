import torch
import torch.nn as nn
from hparams import hparams
from torch.utils.data import Dataset,DataLoader
from dataset import TIMIT_Dataset,my_collect
from model_mapping import DNN_Mapping
import os

if __name__ == "main":

    device = torch.device("cuda:0")

    para = hparams()

    m_model = DNN_Mapping(para)
    m_model = m_model.to(device)
    m_model.train()

    loss_fun = nn.MSELoss()
    loss_fun = loss_fun.to(device)

    optimizer = torch.optim.Adam(
        params = m_model.parameters(),
        lr = para.learning_rate)

    m_Dataset = TIMIT_Dataset(para)
    m_Dataloader = torch.utils.data.DataLoader(m_Dataset,batch_size=para.batch_size,shuffle=True,num_workers=4,collate_fn=my_collect)

    n_epoch = 5
    n_step = 0
    loss_total = 0
    for epoch in range(n_epoch):
        for i_batch,sample_batch in enumerate(m_Dataloader):
            train_X = sample_batch[0]
            train_Y = sample_batch[1]

            train_X = train_X.to(device)
            train_Y = train_Y.to(device)

            output_enh,output_target = m_model(x=train_X,y=train_Y)

            loss = loss_fun(output_enh,output_target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            n_step = n_step+1
            loss_total = loss_total+loss

            if n_step %5 ==0:
                print("epoch = %02d  step = %04d  loss = %.4f"%(epoch,n_step,loss))

            loss_mean = loss_total/n_step
            print("epoch = %2d  mean_loss = %f"%(epoch,loss_mean))
            loss_total = 0
            n_step = 0

            save_name = os.path.join('save','model_%d_%.4f.pth'%(epoch,loss_mean))
            torch.save(m_model,save_name)


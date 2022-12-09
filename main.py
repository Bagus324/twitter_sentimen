from utils.preprocessor import Preprocessor
from utils.preprocessor_lstm import PreprocessorLSTM
from utils.trainer import MultiClassTrainer
from utils.trainer_lstm import ClassificationTrainer

from transformers import BertConfig
import pytorch_lightning as pl


if __name__ =="__main__":
    dataset_binary = "datasets/binary.csv"
    dataset_ite = "datasets/Dataset Twitter Fix - Indonesian Sentiment Twitter Dataset Labeled (1).csv"
    save_dir = "datasets/merged_dataset.pkl"
    pre = PreprocessorLSTM(save_dir, dataset_binary, dataset_ite)
    train_data, val_data, vocab_size, embedding_size = pre.main()

    model = ClassificationTrainer(
        num_layers = 2,
        vocab_size = vocab_size,
        hidden_dim = 256,
        embedding_dim = embedding_size,
        lr = 1e-2
    )

    trainer = pl.Trainer(gpus = 1, max_epochs = 1)
    trainer.fit(model, train_data, val_data)


    # config = BertConfig()


    # model = MultiClassTrainer(
    #     lr = 1e-5,
    #     bert_config = config,
    #     dropout=0.3
    # )
    # trainer = pl.Trainer(gpus = 1, 
    #                     max_epochs = 20, 
    #                     default_root_dir = "./checkpoints/class",
    #                     )

    # trainer.fit(model, datamodule = dm)

    # trainer.predict(model = model, datamodule = dm)
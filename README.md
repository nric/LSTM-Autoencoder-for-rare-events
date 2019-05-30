# LSTM-Autoencoder-for-rare-events
An LSTM (binary classifier) Autoencoder to identify rare events written using TF2.0/Keras

It is written using Tensorflow 2.0/Keras, Pandas and Seaborn libaries.

Since I was playing arround with Autoencoders, I stumbled over this nice mini datascience project
and found it useful to reproduce it to learn to use Autoencoders with LSTM cells:
https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098

The basic idea is:
- Train an LSTM autoencoder to generate/predict "normal" features for the next timestep.
- If the measurment of the next timestep differs greatly from the generated/predicted, this is likely a fault sate and with a certain threshold will be regarded as such. 
- use the few y = 1 data lines from the dataset just for validation and statistics and don't even bother to try to make a superwised classification problem out of this dataset.

I used VS-Code with Iphykernel / jupyter. But I also exported as ipynb if anyone wants to look at it there.

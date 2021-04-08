# Deep learning models with jutpyter notebook

Try [How to forcast](deep-learning/how-to-forecast.ipynb) first.

Then try [modle 16](deep-learning/16.attention-is-all-you-need.ipynb), Attention

# Prediction with Regression topics

## RNN, LSTM, GRU
[Basic model explain](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632)


Among the several ways developed over the years to accurately predict the complex and volatile variation of stock prices, neural networks, more specifically RNNs, have shown significant application on the field. The most common ones are  — LSTM and GRU — .

### Recurrent Neural Network (RNN)
A recurrent neural network (RNN) is a type of artificial neural network designed to recognize data’s sequential patterns to predict the following scenarios. This architecture is especially powerful because of its nodes connections, allowing the exhibition of a temporal dynamic behavior. Another important feature of this architecture is the use of feedback loops to process a sequence. Such a characteristic allows information to persist, often described as a memory. This behavior makes RNNs great for Natural Language Processing (NLP) and time series problems. Based on this structure, architectures called Long short-term memory (LSTM), and Gated recurrent units (GRU) were developed.
An LSTM unit is composed of a cell, an input gate, an output gate, and a forget gate. The cell remembers values over arbitrary time intervals, and the three gates regulate the flow of information into and out of the cell.

![LSTM](https://miro.medium.com/max/700/1*hQeBYYqZa-Qc3ss8q6J0Mg.png)

Source: https://en.wikipedia.org/wiki/Recurrent_neural_network

On the other hand, a GRU has fewer parameters than LSTM, lacking an output gate. Both structures can address the “short-term memory” issue plaguing vanilla RNNs and effectively retain long-term dependencies in sequential data.

![GRU](https://miro.medium.com/max/700/1*8JvGcDVY0tgCVMlpjY2sag.png)
Source: https://en.wikipedia.org/wiki/Recurrent_neural_network

Although LSTM is currently more popular, the GRU is bound to eventually outshine it due to a superior speed while achieving similar accuracy and effectiveness. We are going to see that we have a similar outcome here, and the GRU model also performs better in this scenario.

### [Attention](https://medium.datadriveninvestor.com/attention-in-rnns-321fbcd64f05)
Attention is a mechanism combined in the RNN allowing it to focus on certain parts of the input sequence when predicting a certain part of the output sequence, enabling easier learning and of higher quality. Combination of attention mechanisms enabled improved performance in many tasks making it an integral part of modern RNN networks.

![attention diagram](https://miro.medium.com/max/700/1*wnXVyE8LXPfODvB_Z5vu8A.jpeg)

### [Code example with PyTorch](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632)
Section Model implementation

### Blog [part2](https://medium.com/@shivamakhauri04/reinforcing-a-gaming-agent-lunar-lander-to-land-in-the-landing-pad-7c92c2f9f7c3) 



# Techniques We Can Use for Predicting Stock Prices
As it is a prediction of continuous values, any kind of regression technique can be used:

Linear regression will help you predict continuous values
Time series models are models that can be used for time-related data
ARIMA is one such model that is used for predicting futuristic time-related predictions
LSTM is also one such technique that has been used for stock price predictions. LSTM refers to Long Short Term Memory and makes use of neural networks for predicting continuous values. LSTMs are very powerful and are known for retaining long term memory
However, there is another technique that can be used for stock price predictions which is reinforcement learning.

![Stock Price Reinforcement Learning google stock](https://editor.analyticsvidhya.com/uploads/87552download%20(1).png "image")

# Reenforcemet Learning RL
[Reenforcement regression](https://www.analyticsvidhya.com/blog/2020/10/reinforcement-learning-stock-price-prediction/#:~:text=ARIMA%20is%20one%20such%20model,networks%20for%20predicting%20continuous%20values.)


What is Reinforcement Learning?
Reinforcement learning is another type of machine learning besides supervised and unsupervised learning. This is an agent-based learning system where the agent takes actions in an environment where the goal is to maximize the record. Reinforcement learning does not require the usage of labeled data like supervised learning.

Reinforcement learning works very well with less historical data. It makes use of the value function and calculates it on the basis of the policy that is decided for that action.

Reinforcement learning is modeled as a Markov Decision Process (MDP):

An Environment E and agent states S

A set of actions A taken by the agent

P(s,s’)=>P(st+1=s’|st=s,at=a) is the transition probability from one state s to s’

R(s,s’) – Immediate reward for any action

Why Reinforcement Learning over Supervised approaches (LSTM)?
As described in my other post, Reinforcement learning is a branch of ML which involves taking suitable action to maximize reward in a particular situation. RL differs from the supervised learning in a way that in supervised learning the training data has the answer key with it so the model is trained with the correct answer itself whereas in reinforcement learning, there is no answer but the reinforcement agent decides what to do to perform the given task.


## another blog of tradebot
[article1](https://medium.com/ether-labs/tradebot-stock-trading-using-reinforcement-learning-part1-8b67c9603f33)
[code](https://github.com/shivamakhauri04/TradingBot/blob/master/1_dqn.ipynb)
[article2](https://medium.com/@shivamakhauri04/reinforcing-a-gaming-agent-lunar-lander-to-land-in-the-landing-pad-7c92c2f9f7c3)


# code reference

[LSTM with pytorch](https://www.kaggle.com/taronzakaryan/predicting-stock-price-using-lstm-model-pytorch)

[Q network](https://www.kaggle.com/itoeiji/deep-reinforcement-learning-on-stock-data)


## [How to choose regression model](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/)

### Regression types

#### Linear Regression
It is represented by an equation Y=a+b*X + e, where a is intercept, b is slope of the line and e is error term. This equation can be used to predict the value of target variable based on given predictor variable(s).

![Linear regression](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Linear_Regression1.png)

Important Points:
- There must be linear relationship between independent and dependent variables
- Multiple regression suffers from multicollinearity, autocorrelation, heteroskedasticity.
- Linear Regression is very sensitive to Outliers. It can terribly affect the regression line and eventually the forecasted values.
- Multicollinearity can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable
- In case of multiple independent variables, we can go with forward selection, backward elimination and step wise approach for selection of most significant independent variables.

#### Logistic Regression

Logistic regression is used to find the probability of event=Success and event=Failure. We should use logistic regression when the dependent variable is binary (0/ 1, True/ False, Yes/ No) in nature. Here the value of Y ranges from 0 to 1 and it can represented by following equation.

```
odds= p/ (1-p) = probability of event occurrence / probability of not event occurrence
ln(odds) = ln(p/(1-p))

logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk
```
Above, p is the probability of presence of the characteristic of interest. A question that you should ask here is “why have we used log in the equation?”.

Since we are working here with a binomial distribution (dependent variable), we need to choose a link function which is best suited for this distribution. And, it is logit function. In the equation above, the parameters are chosen to maximize the likelihood of observing the sample values rather than minimizing the sum of squared errors (like in ordinary regression).

logistic regression, logit function, sigmoid function
![logit](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Logistic_Regression.png)

Important Points:
- Logistic regression is widely used for **classification problems**
- Logistic regression doesn’t require linear relationship between dependent and independent variables.  It can handle various types of relationships because it applies a non-linear log transformation to the predicted odds ratio
- To avoid over fitting and under fitting, we **should include all significant variables**. A good approach to ensure this practice is to use a step wise method to estimate the logistic regression
- It requires **large sample sizes** because maximum likelihood estimates are less powerful at low sample sizes than ordinary least square
- The independent variables **should not be correlated with each other** i.e. no multi collinearity.  However, we have the options to include interaction effects of categorical variables in the analysis and in the model.
- If the values of dependent variable is ordinal, then it is called as Ordinal logistic regression
- If dependent variable is multi class then it is known as Multinomial Logistic regression.

#### Polynomial Regression
A regression equation is a polynomial regression equation if the power of independent variable is more than 1. The equation below represents a polynomial equation:
```math
y=a+b*x^2
```
In this regression technique, the best fit line is not a straight line. It is rather a curve that fits into the data points.

![diagram](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Polynomial.png)

Important Points:
- While there might be a temptation to fit a higher degree polynomial to get lower error, this can result in over-fitting. Always plot the relationships to see the fit and focus on making sure that the curve fits the nature of the problem. Here is an example of how plotting can help:
underfitting, overfitting
![diagram2](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/underfitting-overfitting.png)
- Especially look out for curve towards the ends and see whether those shapes and trends make sense. Higher polynomials can end up producing wierd results on extrapolation.

#### Stepwise Regression
This form of regression is used when we deal with multiple independent variables. In this technique, the selection of independent variables is done with the help of an automatic process, which involves no human intervention.

#### Ridge Regression
Ridge Regression is a technique used when the data suffers from multicollinearity (independent variables are highly correlated). In multicollinearity, even though the least squares estimates (OLS) are unbiased, their variances are large which deviates the observed value far from the true value. By adding a degree of bias to the regression estimates, ridge regression reduces the standard errors.

#### Lasso Regression
Similar to Ridge Regression, Lasso (Least Absolute Shrinkage and Selection Operator) also penalizes the absolute size of the regression coefficients. In addition, it is capable of reducing the variability and improving the accuracy of linear regression models.  Look at the equation below: Lasso regression differs from ridge regression in a way that it uses absolute values in the penalty function, instead of squares. This leads to penalizing (or equivalently constraining the sum of the absolute values of the estimates) values which causes some of the parameter estimates to turn out exactly zero. Larger the penalty applied, further the estimates get shrunk towards absolute zero. This results to variable selection out of given n variables.

#### Lasso Regression
lasso regression, l1 regularization

Similar to Ridge Regression, Lasso (Least Absolute Shrinkage and Selection Operator) also penalizes the absolute size of the regression coefficients. In addition, it is capable of reducing the variability and improving the accuracy of linear regression models.  Look at the equation below: Lasso regression differs from ridge regression in a way that it uses absolute values in the penalty function, instead of squares. This leads to penalizing (or equivalently constraining the sum of the absolute values of the estimates) values which causes some of the parameter estimates to turn out exactly zero. Larger the penalty applied, further the estimates get shrunk towards absolute zero. This results to variable selection out of given n variables.
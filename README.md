# IMCprosperity3

## Bayesian Optimization Scripts

This repository contains two scripts, `bayesian.py` and `bayesian2.py`, both utilizing Bayesian optimization techniques but for different use cases. Below, you will find an explanation of each script and how they work.

### bayesian.py - Bayesian Optimization for Backtesting

`bayesian.py` implements Bayesian optimization to fine-tune the parameters of a trading model using backtesting data. The goal of this script is to optimize trading parameters, such as `reversion_beta`, `take_width`, `clear_width`, and `adverse_volume`, by evaluating their effect on the model's performance using backtesting.

#### Features:
- **Bayesian Optimization**: Utilizes Bayesian optimization to explore the parameter space and find the best combination of parameters that maximize the total profit from the backtest.
- **Optimization Parameters**: The optimization focuses on the following parameters:
  - `reversion_beta`: Controls the reversion speed of the model.
  - `take_width`: The width of the take positions.
  - `clear_width`: The width of the clear positions.
  - `adverse_volume`: The volume limit for adverse movements.
- **Backtesting**: A simulated backtest is run for each set of parameters, and the total profit is recorded as the objective function for optimization.
- **Optimization Process**: The script uses Bayesian optimization to iteratively adjust the parameters and maximize the total profit over multiple trials.

#### How To Run:
1. Clone the repository:  
    ```bash
    git clone https://github.com/DasinuV0/IMCprosperity3.git
    ```
2. Create a clean new virtual environment (i use pipenv):    
    ```bash
3. Install all the required packages:  
    ```bash
    pip install -r requirements.txt
    ```
4. Go to the `imc-prosperity-3-backtester` directory:  
    ```bash
    cd ./imc-prosperity-3-backtester/
    ```
5. Install **imc-prosperity-3-submitter** in editable mode:  
    ```bash
    pip install -e . --config-settings editable=true
    ```
6. Run `bayesian.py`:  
    ```bash
    python ..\bayesian.py
    ```

### bayesian2.py - Bayesian Optimization for Website Data

`bayesian2.py` applies Bayesian optimization to data collected from a website. This data could be related to market conditions, customer behavior, or any other relevant dataset that impacts the model. The goal is to find the best set of parameters based on the optimization results from the website data.

#### Features:
- **Bayesian Optimization**: Just like `bayesian.py`, this script uses Bayesian optimization, but it focuses on real-world data from the website.
- **Optimization Parameters**: Similar to `bayesian.py`, but the parameters could vary depending on the context of the website data. These parameters are dynamically adjusted to optimize the model's performance based on the website's metrics.
- **Website Data**: Instead of using backtest data, the script pulls data from a website and uses it to inform the optimization process.
- **Optimization Process**: Bayesian optimization adjusts the model's parameters to maximize a performance metric derived from the website's data.

#### How To Run:
1. Clone the repository:  
    ```bash
    git clone https://github.com/DasinuV0/IMCprosperity3.git
    ```
2. Create a clean new virtual environment (i use pipenv):    
    ```bash
    pipenv shell
3. Install all the required packages:  
    ```bash
    pip install -r requirements.txt
    ```
4. Go to the `imc-prosperity-3-submitter` directory:  
    ```bash
    cd ./imc-prosperity-3-submitter/
    ```
5. Install **imc-prosperity-3-submitter** in editable mode:  
    ```bash
    pip install -e . --config-settings editable=true
    ```
6. Run `bayesian2.py`:  
    ```bash
    python ..\bayesian2.py
    ```

## Conclusion

- **bayesian.py** is designed to optimize trading model parameters using backtesting data, focusing on maximizing total profit.
- **bayesian2.py** extends the concept of Bayesian optimization by using website data to optimize parameters for improving performance metrics based on real-world inputs.

These scripts are designed to demonstrate the flexibility and power of Bayesian optimization in different contexts. By applying this technique, both scripts can help you fine-tune complex models to achieve better performance and results.

# COM724_AE2
 Crypto insights and forecaster


# Project setup and configuration

Dependencies installed with yarn

```
npm create vite@latest crypto-forecaster -- --template react-ts
cd crypto-forecaster
yarn install
```

```
 yarn add recharts @types/recharts lucide-react 
```

material UI for the Cards
```
yarn add @mui/material @emotion/react @emotion/styled
```

for xgboost with PyCaret:

downgrade scipy

``` pip install scipy==1.10.1 --force-reinstall ```

``` conda install xgboost ```

for Prophet with PyCaret:

``` pip install matplotlib==3.7.1 --force-reinstall ```

``` pip install numpy==1.24 --force-reinstall ```

## oustanding work

Integrate with Real-Time Data: Modify the code to pull real-time data from a cryptocurrency API

Check best practices for 7 and 30 day moving average plots

Feature importance for extracted features

outline the location of the analysis work from the notebooks, with links to each

installation and running instructions for the project.
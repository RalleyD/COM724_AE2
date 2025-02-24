import React, { useEffect, useState } from "react";
import { formatCurrency } from "../utils/formatters";
import { CryptoData } from '../props/types';
import {
  BarChart, 
  Bar, 
  LineChart,
  Line,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Cell
} from 'recharts';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import Box from '@mui/material/Box';
import Alert from '@mui/material/Alert';

interface CryptoChartProps {
  cryptoData: CryptoData[];
}

type ChartType = 'bar' | 'line';

const CryptoChart: React.FC<CryptoChartProps> = ({ cryptoData }) => {
  // Prepare simplified data for chart
  console.log("Chart data received:", cryptoData);

  const [chartType, setChartType] = useState<chartType>('bar');
  const [processedData, setProcessedData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  // colurs for the bars/lines
  const COLOURS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042',
                  '#0088FE', '#00C49F', '#FFBB28', '#FF8042'];
  
  // process data when crypto changes 
  useEffect( () => {
    try {
      console.log("Raw crypto data:", cryptoData);
      if(!cryptoData || !Array.isArray(cryptoData) || cryptoData.length == 0) {
        setError("No data available to display!");
        setProcessedData([]);
        return;
      }

      // take top 10 cryptos
      const topCoins = cryptoData.slice(0 ,10);

      // transform data for chart display
      const transformedData = topCoins.map(crypto => ({
        name: crypto.symbol.toUpperCase(),
        price: crypto.current_price,
        marketCap: crypto.market_cap / 1000000000, // convert to billions
        change24h: crypto.price_change_percentage_24h,
        id: crypto.id,
      }));
      console.log("Transformed chart data:", transformedData);

      setProcessedData(transformedData);
      setError(null);
    } catch (err) {
      console.error("Error processing chart data:", err);
      setError("Failed to process chart data");
      setProcessedData([]);
    }
  }, [cryptoData]);

  const chartData = cryptoData.slice(0, 10).map(crypto => ({
    name: crypto.symbol.toUpperCase(),
    price: crypto.current_price,
    marketCap: crypto.market_cap / 1000000000, // Convert to billions for better display
    change24h: crypto.price_change_percentage_24h
  }));

  console.log("Transformed chart data:", chartData);

  // Toggle between chart types
  const handleChartTypeChange = (type: ChartType) => {
    setChartType(type);
  };

  // Render the actual chart based on type
  const renderChart = () => {
    if (processedData.length === 0) {
      return (
        <Box 
          display="flex" 
          justifyContent="center" 
          alignItems="center" 
          height="100%"
          flexDirection="column"
        >
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No data available
          </Typography>
          {error && (
            <Alert severity="error" style={{ maxWidth: "80%" }}>
              {error}
            </Alert>
          )}
        </Box>
      );
    }
    
    // Check if we have valid numerical data
    const hasValidData = processedData.some(item => 
      !isNaN(item.price) && !isNaN(item.marketCap)
    );
    
    if (!hasValidData) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height="100%">
          <Alert severity="warning">
            Data contains non-numeric values. Check console for details.
          </Alert>
        </Box>
      );
    }
    
    if (chartType === 'bar') {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={processedData}
            margin={{ top: 20, right: 30, left: 60, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis 
              yAxisId="left"
              orientation="left"
              stroke="#8884d8"
              label={{ value: 'Price (USD)', angle: -90, position: 'insideLeft' }}
              tickFormatter={(value) => formatCurrency(value)}
            />
            <YAxis 
              yAxisId="right"
              orientation="right"
              stroke="#82ca9d"
              label={{ value: 'Market Cap (Billions)', angle: 90, position: 'insideRight' }}
            />
            <Tooltip 
              formatter={(value, name) => {
                if (name === "price") return [formatCurrency(Number(value)), "Price"];
                if (name === "marketCap") return [`$${Number(value).toFixed(2)}B`, "Market Cap"];
                return [value, name];
              }}
            />
            <Legend />
            <Bar yAxisId="left" dataKey="price" name="Price (USD)" fill="#8884d8">
              {processedData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLOURS[index % COLOURS.length]} />
              ))}
            </Bar>
            <Bar yAxisId="right" dataKey="marketCap" name="Market Cap (Billions)" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
      );
    } else {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={processedData}
            margin={{ top: 20, right: 30, left: 60, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis 
              tickFormatter={(value) => formatCurrency(value)}
              label={{ value: 'Price (USD)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip formatter={(value) => formatCurrency(Number(value))} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#8884d8" 
              activeDot={{ r: 8 }}
              name="Price (USD)"
            />
          </LineChart>
        </ResponsiveContainer>
      );
    }
  };
  
  return (
    <Card>
      <CardHeader 
        title={
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant='h5' component='div'>Top 10 Cryptocurrencies</Typography>
            <ButtonGroup variant="outlined" size="small">
              <Button 
                onClick={() => handleChartTypeChange('bar')}
                variant={chartType === 'bar' ? 'contained' : 'outlined'}
              >
                Bar
              </Button>
              <Button 
                onClick={() => handleChartTypeChange('line')}
                variant={chartType === 'line' ? 'contained' : 'outlined'}
              >
                Line
              </Button>
            </ButtonGroup>
          </Box>
        }
      />
      <CardContent>
        <div style={{ height: "400px", width: "100%" }}>
          {renderChart()}
        </div>
        
        {/* Debugging section - remove in production */}
        <Box mt={2} p={2} border="1px dashed #ccc" borderRadius={1} display="none">
          <Typography variant="subtitle2" gutterBottom>Debug Information</Typography>
          <Typography variant="body2">Data points: {processedData.length}</Typography>
          <Typography variant="body2">First item: {processedData.length > 0 ? JSON.stringify(processedData[0]) : 'None'}</Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default CryptoChart;
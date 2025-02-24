import React, { useEffect, useState } from "react";
import { AlertCircle, CircleAlertIcon, RefreshCw } from "lucide-react";
import { formatCurrency, formatPercentage } from "../utils/formatters";
import { useCryptoData } from "../hooks/useCryptoData";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { CryptoData, TimeRange } from '../props/types';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card'
import CardHeader from '@mui/material/CardHeader'
/* 3rd define a DashboardState interface to type our component's state */

/* 4th Core component - type the component as a Function Component */
const CryptoDashboard : React.FC = () => {
    /* 5th - state management with TypeScript
    generics i.e useState which conforms to the DashboardState
    interface.
    Now, TypeScript will enforce these types throughout the component.
    */
    const [selectedTimeRange, setSelectedTimeRange] = useState<TimeRange>('24h');

    const { 
        data: cryptoData, 
        loading, 
        error, 
        refresh 
      } = useCryptoData({
        timeRange: selectedTimeRange,
        refreshInterval: 60000, // 1 minute refresh
        limit: 30 // Top 30 cryptocurrencies
      });

    const handleTimeRangeChange = (timeRange: TimeRange) => {
        setSelectedTimeRange(timeRange);
    };

    /* 7th type safe component rendering */
    const renderTopPerformers = (cryptos : CryptoData[]): JSX.Element => (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {cryptos.slice(0,3).map((crypto) => (
                <div key={crypto.id} className="p-4 border rounded-lg">
                <div className="flex items-center space-x-2">
                  <img 
                    src={crypto.image} 
                    alt={crypto.name} 
                    className="w-8 h-8"
                  />
                  <h3 className="font-bold">{crypto.name}</h3>
                </div>
                <div className="mt-2 space-y-1">
                  <p>Price: {formatCurrency(crypto.current_price)}</p>
                  <p className={crypto.price_change_percentage_24h >= 0 ? 'text-green-600' : 'text-red-600'}>
                    24h Change: {formatPercentage(crypto.price_change_percentage_24h)}
                  </p>
                  <p>Market Cap: {formatCurrency(crypto.market_cap)}</p>
                </div>
              </div>
            ))}
        </div>
    );

    if (loading) {
        return (
        <div className="flex items-center justify-center h-64 text-red-500">Loading...</div>
        );
    }

    if (error) {
        return (
        <div className="flex items-center justify-center h-64 text-red-500">
            <AlertCircle className="mr-2" />
            Error:
            {error}
            <button 
                onClick={refresh} 
                className="ml-4 p-2 bg-blue-500 text-white rounded-md flex items-center"
            >
          <RefreshCw className="w-4 h-4 mr-2" /> Try Again
        </button>
        </div>
    );}

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold">Market Overview</h2>
                <div className="flex space-x-2">
                    <button 
                        onClick={() => handleTimeRangeChange('24h')}
                        className={`px-3 py-1 rounded-md ${selectedTimeRange === '24h' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
                    >
                        24h
                    </button>
                    <button 
                        onClick={() => handleTimeRangeChange('7d')}
                        className={`px-3 py-1 rounded-md ${selectedTimeRange === '7d' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
                    >
                        7d
                    </button>
                    <button 
                        onClick={() => handleTimeRangeChange('30d')}
                        className={`px-3 py-1 rounded-md ${selectedTimeRange === '30d' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
                    >
                        30d
                    </button>
                    <button 
                        onClick={() => handleTimeRangeChange('90d')}
                        className={`px-3 py-1 rounded-md ${selectedTimeRange === '90d' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
                    >
                        90d
                    </button>
                    <button 
                        onClick={refresh}
                        className="p-1 rounded-full bg-gray-100 hover:bg-gray-200"
                        title="Refresh data"
                    >
                        <RefreshCw className="w-5 h-5" />
                    </button>
                </div>
        </div>

          <Card>
            <CardHeader>
              <Typography variant='h5' component='div'>Top 30 Cryptocurrencies</Typography>
            </CardHeader>
            <CardContent>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cryptoData.slice(0, 5)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis 
                        domain={['auto', 'auto']}
                        tickFormatter={(value) => formatCurrency(value)}
                        />
                        <Tooltip 
                        formatter={(value: number) => [formatCurrency(value), 'Price']}
                        />
                        <Legend />
                        <Line 
                        type="monotone" 
                        dataKey="current_price" 
                        stroke="#8884d8" 
                        name="Price (USD)"
                        dot={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
    
          <Card>
            <CardHeader>
              <Typography variant='h5' component='div'>Top Performers</Typography>
            </CardHeader>
            <CardContent>
              {renderTopPerformers(cryptoData)}
            </CardContent>
          </Card>
        </div>
    );
};
    
export default CryptoDashboard;
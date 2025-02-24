import React, { useState, useEffect } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { formatCurrency, formatPercentage } from "../utils/formatters";
import { useCryptoData } from "../hooks/useCryptoData";
import { CryptoData, TimeRange } from '../props/types';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardHeader from '@mui/material/CardHeader';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import CryptoGrid from "./CryptoGrid";
import CryptoChart from './CryptoChart';

const CryptoDashboard: React.FC = () => {
    const [selectedTimeRange, setSelectedTimeRange] = useState<TimeRange>('24h');
    const [displayType, setDisplayType] = useState<'chart' | 'grid'>('chart'); // Default to chart view
    const [dataLoaded, setDataLoaded] = useState<boolean>(false);

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

    // Check if data is loaded successfully
    useEffect(() => {
        if (cryptoData && cryptoData.length > 0) {
            setDataLoaded(true);
            console.log("Crypto data loaded:", cryptoData.length, "items");
        } else {
            setDataLoaded(false);
        }
    }, [cryptoData]);

    const handleTimeRangeChange = (timeRange: TimeRange) => {
        setSelectedTimeRange(timeRange);
    };

    const renderTopPerformers = (cryptos: CryptoData[]): JSX.Element => (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {cryptos.slice(0, 3).map((crypto) => (
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
            <div className="flex items-center justify-center h-64">
                <Typography variant="h6">Loading cryptocurrency data...</Typography>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center justify-center h-64 text-red-500">
                <AlertCircle className="mr-2" />
                <Typography variant="body1" color="error">
                    Error: {error}
                </Typography>
                <Button 
                    onClick={refresh} 
                    variant="contained"
                    color="primary"
                    startIcon={<RefreshCw />}
                    style={{ marginLeft: '16px' }}
                >
                    Try Again
                </Button>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <Typography variant="h4" component="h2">Market Overview</Typography>
                <ButtonGroup variant="outlined" size="small">
                    <Button 
                        onClick={() => handleTimeRangeChange('24h')}
                        variant={selectedTimeRange === '24h' ? 'contained' : 'outlined'}
                    >
                        24h
                    </Button>
                    <Button 
                        onClick={() => handleTimeRangeChange('7d')}
                        variant={selectedTimeRange === '7d' ? 'contained' : 'outlined'}
                    >
                        7d
                    </Button>
                    <Button 
                        onClick={() => handleTimeRangeChange('30d')}
                        variant={selectedTimeRange === '30d' ? 'contained' : 'outlined'}
                    >
                        30d
                    </Button>
                    <Button 
                        onClick={() => handleTimeRangeChange('90d')}
                        variant={selectedTimeRange === '90d' ? 'contained' : 'outlined'}
                    >
                        90d
                    </Button>
                    <Button 
                        onClick={refresh}
                        title="Refresh data"
                    >
                        <RefreshCw style={{ width: '18px', height: '18px' }} />
                    </Button>
                </ButtonGroup>
            </div>

            {/* View toggle buttons */}
            <div className="flex justify-center mb-4">
                <ButtonGroup variant="outlined">
                    <Button
                        onClick={() => setDisplayType('chart')}
                        variant={displayType === 'chart' ? 'contained' : 'outlined'}
                    >
                        Chart View
                    </Button>
                    <Button
                        onClick={() => setDisplayType('grid')}
                        variant={displayType === 'grid' ? 'contained' : 'outlined'}
                    >
                        Grid View
                    </Button>
                </ButtonGroup>
            </div>

            {displayType === 'chart' ? (
                <>
                    {dataLoaded ? (
                        <CryptoChart cryptoData={cryptoData} />
                    ) : (
                        <Card>
                            <CardContent>
                                <Typography variant="body1" align="center">
                                    No chart data available. Please try refreshing.
                                </Typography>
                            </CardContent>
                        </Card>
                    )}
                    
                    <Card>
                        <CardHeader>
                            <Typography variant='h5' component='div'>Top Performers</Typography>
                        </CardHeader>
                        <CardContent>
                            {renderTopPerformers(cryptoData)}
                        </CardContent>
                    </Card>
                </>
            ) : (
                <Card>
                    <CardHeader>
                        <Typography variant='h5' component='div'>All Cryptocurrencies (6x5 Grid)</Typography>
                    </CardHeader>
                    <CardContent>
                        <CryptoGrid cryptoData={cryptoData} />
                    </CardContent>
                </Card>
            )}
        </div>
    );
};
    
export default CryptoDashboard;
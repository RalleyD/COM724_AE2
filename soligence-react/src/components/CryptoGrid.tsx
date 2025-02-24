/*
dedicated component for the crypto overview

separation of concerns from the dashboard

enhances maintainability and reusability
*/
import React from 'react';
import { CryptoData } from '../props/types';
import { formatCurrency, formatPercentage } from '../utils/formatters';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';


// component props
interface CryptoGridProps {
    cryptoData : CryptoData[];
}

// component implementation
const CryptoGrid: React.FC<CryptoGridProps> = ({ cryptoData }) => {
    return (
    <div className="crypto-grid-container">
        {/* CSS Grid container
        activates CSS grid layout
        creates 6 equal width columns
        creates 5 equal height rows
        add spacing between grid items.
        */}
        <div 
        style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(6, 1fr)', // 6 columns
            gridTemplateRows: 'repeat(5, 1fr)',     // 5 rows
            gap: '16px'
        }}
        >
        {/* Map through all 30 crypto items */}
        {cryptoData.slice(0, 30).map((crypto) => (
            <Card 
            key={crypto.id} 
            style={{ 
                height: '100%',
                display: 'flex',
                flexDirection: 'column'
            }}
            >
            <CardContent>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                <img 
                    src={crypto.image} 
                    alt={crypto.name} 
                    style={{ width: '24px', height: '24px', marginRight: '8px' }}
                />
                <Typography variant="subtitle1" component="div" style={{ fontWeight: 'bold' }}>
                    {crypto.name}
                </Typography>
                <Typography 
                    variant="body2" 
                    color="text.secondary" 
                    style={{ marginLeft: '4px' }}
                >
                    {crypto.symbol.toUpperCase()}
                </Typography>
                </div>
                
                <Typography variant="h6" component="div" style={{ marginBottom: '4px' }}>
                {formatCurrency(crypto.current_price)}
                </Typography>
                
                <Typography 
                variant="body2" 
                component="div"
                style={{ 
                    color: crypto.price_change_percentage_24h >= 0 ? '#16a34a' : '#dc2626',
                    marginBottom: '4px'
                }}
                >
                {formatPercentage(crypto.price_change_percentage_24h)} (24h)
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                Market Cap: {formatCurrency(crypto.market_cap)}
                </Typography>
            </CardContent>
            </Card>
        ))}
        </div>
    </div>
    );
};

export default CryptoGrid;
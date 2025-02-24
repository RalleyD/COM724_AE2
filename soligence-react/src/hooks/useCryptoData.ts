import { useState, useEffect } from 'react';
import { CryptoData, TimeRange } from '../props/types';

interface UseCryptoDataProps {
    timeRange?: TimeRange;
    refreshInterval?: number;
    limit?: number;
}

interface UseCryptoDataReturn {
    data: CryptoData[];
    loading: boolean;
    error: string | null;
    refresh: () => Promise<void>;
}

export const useCryptoData = ({
    timeRange = '24h',
    refreshInterval = 240000,
    limit = 30
}: UseCryptoDataProps = {}): UseCryptoDataReturn => {
    const [data, setData] = useState<CryptoData[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const fetchData = async (): Promise<void> => {
        try {
          const response = await fetch(
            `https://api.coingecko.com/api/v3/coins/markets?` +
            `vs_currency=usd&order=market_cap_desc&per_page=${limit}&page=1&` +
            `sparkline=true&price_change_percentage=${timeRange}`
          );
    
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
    
          const jsonData: CryptoData[] = await response.json();
          setData(jsonData);
          setError(null);
        } catch (err) {
          setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
          setLoading(false);
        }
      };
    
      useEffect(() => {
        fetchData();
    
        if (refreshInterval > 0) {
          const interval = setInterval(fetchData, refreshInterval);
          return () => clearInterval(interval);
        }
      }, [timeRange, limit, refreshInterval]);
    
      return {
        data,
        loading,
        error,
        refresh: fetchData
    };
};
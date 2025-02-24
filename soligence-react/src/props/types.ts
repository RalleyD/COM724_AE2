/*
Types definition that are used in the components
and provide structure to our data

This all helps TypeScript understand what data
to expect

n.b update interfaces when API changes/updates
*/

export interface CryptoData {
    /* 1st: define the shape of the crypto data
       we expect from the API
       Properties:
    */
    id: string;
    name: string;
    symbol: string;
    current_price: number;
    market_cap: number;
    market_cap_rank: number;
    price_change_percentage_24h: number;
    sparkline_in_7d: {
        price: number[];
    };
    total_volume: number;
    image: string;
    last_updated: string;
}

/* define a type range using a union type, only these values are allowed
   and only one can exist at one time */
export type TimeRange = '24h' | '7d' | '30d' | '90d';

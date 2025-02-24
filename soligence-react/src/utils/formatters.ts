/* 2nd formatter functions for consistent data presentation */

export const formatCurrency = (value: number): string => {
    /**! format number as a currency (USD)
     *   uses TypeScripts annotation (: string)
     *   to ensure a string is returned
     * @param value [number] : input number to format
     * 
     * @returns string : formatted number as USD
    */
    // Intl - canonical local names for localisation
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value);
};

export const formatPercentage = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value/100);
};

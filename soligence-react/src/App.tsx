import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import CryptoDashboard from './components/CryptoDashboard'

function App() {
  return (
    <div className='container mx-auto p-4'>
      <h1 className='text-2xl font-bold mb-4'>
        Crypto Insights and Forecaster
      </h1>
      <CryptoDashboard />
    </div>
  )
}

export default App

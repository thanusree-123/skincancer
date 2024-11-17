import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';  // If you have global styles, otherwise remove this line
import App from './App';
import reportWebVitals from './reportWebVitals';

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')  // Links to the div in index.html
);

// Optional: For performance monitoring, can be skipped if not needed
reportWebVitals();
 

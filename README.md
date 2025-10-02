# Portfolio Dashboard

A comprehensive portfolio management and risk analysis platform built with FastAPI and modern web technologies.

## Features

### 📊 Portfolio Management
- **Real-time Portfolio Tracking**: Monitor your investments with live data
- **Multi-Broker Support**: Import data from Merrill Edge, Fidelity, and other brokers
- **Risk Analysis**: Comprehensive Greeks analysis, concentration risk, and VaR
- **Cash Management**: CSP cash allocation and cash balance tracking

### 🔍 Advanced Analytics
- **Portfolio Overview**: Total value, P&L, position counts
- **Stock Holdings**: Detailed stock analysis and metrics
- **Options Holdings**: Options-specific metrics and Greeks
- **Concentration Analysis**: HHI index, top positions, diversification metrics

### 🎯 User Interface
- **Modern Dashboard**: Clean, responsive web interface
- **Advanced Filtering**: Filter by ticker, position type, and more
- **Smart Sorting**: Sort by value, P&L, quantity, and other metrics
- **Real-time Updates**: Live data refresh and portfolio updates

## Project Structure

```
Portfolio_Dashboard/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── portfolio_management/     # Core portfolio logic
│   ├── portfolio_api.py      # FastAPI application
│   ├── csv_parser.py         # Multi-broker CSV parsing
│   ├── merrill_edge_parser.py # Merrill Edge specific parser
│   └── csp_cash_allocator.py # Cash Secured Put analysis
├── web_interface/            # Frontend components
│   ├── templates/            # HTML templates
│   └── static/              # CSS, JS, images
├── shared/                   # Shared utilities
│   ├── config.py            # Configuration settings
│   └── base.py              # Base classes and utilities
└── data/                    # Database schemas and data
    └── database_schema.sql  # Supabase database schema
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment** (Optional):
   Create a `.env` file with your API keys for full functionality:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   POLYGON_API_KEY=your_polygon_key
   ```
   
   **Note**: The dashboard works without these credentials using in-memory storage and mock data for demonstration purposes.

3. **Set up Database** (Optional):
   If using Supabase, run the SQL schema in your Supabase dashboard:
   ```bash
   cat data/database_schema.sql
   ```

4. **Start the Dashboard**:
   ```bash
   python main.py
   ```
   
   Or use the provided scripts:
   ```bash
   # Windows
   run.bat
   
   # PowerShell
   .\run.ps1
   ```

5. **Access the Dashboard**:
   Open your browser to `http://localhost:8000`

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run with coverage report
python run_tests.py --coverage

# Run specific test file
python run_tests.py --test tests/test_portfolio_api.py
```

## Usage

### Importing Portfolio Data
1. Export your portfolio data from your broker (CSV format)
2. Use the "Import CSV" button in the dashboard
3. The system will automatically detect the broker format and parse your data

### Supported Brokers
- **Merrill Edge**: Full support with option ticker parsing
- **Fidelity**: Basic CSV import support
- **Generic CSV**: Standard format support

### Portfolio Analysis
- View real-time portfolio value and P&L
- Analyze risk metrics and Greeks
- Monitor cash allocation for CSPs
- Track concentration and diversification

## API Endpoints

- `GET /` - Main dashboard interface
- `GET /api/portfolio/positions` - Get all positions
- `POST /api/portfolio/positions` - Add new position
- `DELETE /api/portfolio/positions/{id}` - Delete position
- `POST /api/portfolio/import` - Import CSV data
- `GET /api/portfolio/risk-metrics` - Get risk analysis
- `GET /api/portfolio/export` - Export portfolio data

## Recent Improvements

### ✨ New Features
- **Enhanced Options Analysis**: Complete Greeks calculation engine with Black-Scholes model
- **Smart Caching**: 1-minute cache for risk calculations to improve performance
- **Fallback Mode**: Works without database credentials using in-memory storage
- **Real-time Updates**: Auto-refresh dashboard every 5 minutes with manual refresh button
- **Better Error Handling**: Comprehensive logging and graceful error recovery
- **Responsive UI**: Improved mobile-friendly design with status indicators

### 🔧 Technical Improvements
- **Missing Dependencies**: Fixed all import issues and missing analyzer classes
- **Performance Optimization**: Added caching layer and optimized database queries
- **Comprehensive Testing**: Full test suite with 90%+ coverage
- **Better Documentation**: Updated README with clear setup instructions
- **Code Quality**: Improved error handling, logging, and code structure

### 🎯 UI Enhancements
- **Refresh Button**: Manual refresh with loading indicators
- **Status Indicator**: Real-time connection status
- **Loading States**: Better user feedback during data loading
- **Error Messages**: Clear error reporting with auto-dismiss
- **Responsive Design**: Works seamlessly on desktop and mobile

## Development

### Running in Development Mode
```bash
python main.py
```

### Code Style
The project uses Black for code formatting:
```bash
black .
```

### Testing
```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for personal use and educational purposes.

## Support

For issues and questions, please check the documentation or create an issue in the repository.

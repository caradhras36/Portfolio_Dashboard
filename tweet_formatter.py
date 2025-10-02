"""Module for formatting market data into tweets."""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple, Union, Any
from config import TICKER, POPULAR_STOCKS, HIGH_IV_WATCHLIST
import logging
import pytz
import platform
import random

logger = logging.getLogger(__name__)

class TweetFormatter:
    """Formats market data into tweets."""
    
    EMOJI_CALENDAR = '📅'
    EMOJI_SPY = '📈'
    EMOJI_PC = '⚖️'
    EMOJI_VIX = '😱'
    EMOJI_BOND = '📊'
    EMOJI_SECTOR = '🏢'
    EMOJI_UP = '👍'
    EMOJI_DOWN = '👎'
    EMOJI_FG_EXTREME_FEAR = '🥶'
    EMOJI_FG_FEAR = '😨'
    EMOJI_FG_NEUTRAL = '😐'
    EMOJI_FG_GREED = '🤑'
    EMOJI_FG_EXTREME_GREED = '🤩'
    EMOJI_FG_UNKNOWN = '❓'

    def _get_fg_emoji(self, rating: str) -> str:
        """Get the emoji for a given Fear & Greed rating."""
        rating = rating.lower()
        if "extreme greed" in rating:
            return self.EMOJI_FG_EXTREME_GREED
        elif "greed" in rating:
            return self.EMOJI_FG_GREED
        elif "neutral" in rating:
            return self.EMOJI_FG_NEUTRAL
        elif "extreme fear" in rating:
            return self.EMOJI_FG_EXTREME_FEAR
        elif "fear" in rating:
            return self.EMOJI_FG_FEAR
        return self.EMOJI_FG_UNKNOWN

    @staticmethod
    def _format_hourly_data(hourly_data: List[float]) -> List[str]:
        """Formats hourly data into one or two lines for a tweet."""
        if not hourly_data:
            return []
        hourly_prices_formatted = []
        prev_price = None
        for i, price in enumerate(hourly_data):
            hour = 10 + i  # Market opens at 9:30 AM ET, first hour is 10 AM
            am_pm = "AM" if hour < 12 else "PM"
            hour_12 = hour if hour <= 12 else hour - 12
            change_str = ""
            if prev_price is not None:
                change = ((price - prev_price) / prev_price) * 100
                change_str = f" ({change:+.1f}%)"
            hourly_prices_formatted.append(f"{hour_12}{am_pm}: ${price:.2f}{change_str}")
            prev_price = price
        
        if len(hourly_prices_formatted) > 4:
            mid = (len(hourly_prices_formatted) + 1) // 2 # Ensure first part is not shorter if odd
            return [
                f"⏰ Hourly (1): {' | '.join(hourly_prices_formatted[:mid])}",
                f"⏰ Hourly (2): {' | '.join(hourly_prices_formatted[mid:])}"
            ]
        else:
            return [f"⏰ Hourly: {' | '.join(hourly_prices_formatted)}"]

    @staticmethod
    def format_metric_explanations() -> Tuple[str, str]:
        """Format metric explanations into a tweet string and return hashtags separately."""
        hashtag_line = "\n#Investing #Finance #StockMarket #MarketAnalysis #FinancialLiteracy"
        lines = [
            "📚 Metrics Explained (5/5):",
            "• $SPX: S&P 500 Index",
            "• $VIX: Market Volatility Index",
            "• $MOVE: Bond Market Volatility",
            "• $CPC: Put/Call Ratio",
            "• F&G: Fear & Greed Index",
            "• IV: Implied Volatility (Rank, Spikes, Crushes)",
            "• Sectors: S&P 500 Sectors Performance",
            "• Breadth: Advancers/Decliners",
            "",
            "💡 Want different metrics or specific stock IV analysis? Let me know in the comments!",
            hashtag_line
        ]
        tweet_content = "\n".join(lines)
        # Return the hashtags without the leading newline for direct appending later
        return tweet_content, hashtag_line.lstrip()

    def get_trading_day_header(self) -> str:
        """Return a human-friendly trading day header for the US market."""
        eastern = pytz.timezone('US/Eastern')
        now_utc = datetime.now(timezone.utc)
        now_eastern = now_utc.astimezone(eastern)
        # If after 4pm ET, use today; if before, use previous weekday
        if now_eastern.hour < 16:
            # If before 4pm ET, use previous weekday
            trading_day = now_eastern - timedelta(days=1)
            # If it's Monday, go back to Friday
            if trading_day.weekday() == 6:  # Sunday
                trading_day -= timedelta(days=2)
            elif trading_day.weekday() == 5:  # Saturday
                trading_day -= timedelta(days=1)
        else:
            trading_day = now_eastern
        # Format: Monday, May 6, 2025 - Choose format string based on OS
        if platform.system() == "Windows":
             date_str = trading_day.strftime('%A, %B %#d, %Y')
        else: # Linux, macOS, etc.
             date_str = trading_day.strftime('%A, %B %-d, %Y')
        return f"📅 Market Report for Trading Day: {date_str}"

    def format_single_tweet(
        self,
        pcr_data: Optional[float],
        vix_data: Dict[str, float],
        fear_greed_data: Dict[str, Any],
        popular_stocks_iv: List[Tuple[str, float, float]],
        high_vol_stocks_iv: List[Tuple[str, float, float]]
    ) -> List[str]:
        """Formats the market data into an engaging, question-led tweet thread."""
        
        # Combine and sort all stocks by IV Rank to find top opportunities
        all_stocks = []
        if popular_stocks_iv:
            all_stocks.extend(popular_stocks_iv)
        if high_vol_stocks_iv:
            # Filter out stocks with low IV
            filtered_high_vol = [
                (t, r, i) for t, r, i in high_vol_stocks_iv
                if i > 1 or t in HIGH_IV_WATCHLIST
            ]
            all_stocks.extend(filtered_high_vol)
        
        # Remove duplicates and sort by IV Rank (highest first)
        seen_tickers = set()
        unique_stocks = []
        for ticker, iv_rank, current_iv in all_stocks:
            if ticker not in seen_tickers:
                seen_tickers.add(ticker)
                unique_stocks.append((ticker, iv_rank, current_iv))
        
        unique_stocks.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 opportunities
        top_3 = unique_stocks[:3] if len(unique_stocks) >= 3 else unique_stocks
        
        # Random attention-grabbing hooks
        attention_hooks = [
            "Day's most important tweet for option sellers is here",
            "Today's must-read for options premium sellers",
            "Critical intel for options sellers today",
            "Today's premium selling goldmine is here",
            "Options sellers: This is your daily edge",
            "Today's options premium opportunities are live",
            "Daily options seller's treasure trove is here"
        ]
        
        # Random hook questions
        hook_questions = [
            "If you could sell options premium on ONLY ONE stock today, which would you pick?",
            "Which stock would you sell options premium on RIGHT NOW?",
            "Pick ONE ticker for options premium selling today - which one?",
            "If you had to choose just ONE stock for options premium selling, what would it be?",
            "Which of these high IV stocks catches your eye for options premium selling?",
            "Time to pick your options premium selling target - which one?",
            "If you could only sell options premium on ONE stock today, which would you choose?"
        ]
        
        # Random intro lines
        intro_lines = [
            "Here are the top 3 IV opportunities right now:",
            "Check out these 3 options premium-selling goldmines:",
            "These 3 stocks are screaming for options premium sellers:",
            "Top 3 IV opportunities waiting for you:",
            "Here's your options premium-selling hit list:",
            "3 stocks with juicy IV ranks right now:",
            "Your top 3 options premium-selling targets:"
        ]
        
        # Build the engaging tweet
        lines = [
            random.choice(attention_hooks),
            "",
            random.choice(hook_questions),
            "",
            random.choice(intro_lines),
            ""
        ]
        
        # Add top 3 stocks with engaging descriptions
        iv_descriptions = [
            "Options premiums are 🔥",
            "Volatility explosion",
            "Options premium dream",
            "Premium selling paradise",
            "Options seller's market",
            "IV goldmine",
            "Options premium jackpot",
            "Volatility feast",
            "Options premium treasure",
            "High IV heaven",
            "Options premium bonanza",
            "Volatility gold",
            "Options premium goldmine",
            "Premium selling feast",
            "IV treasure trove"
        ]
        
        for i, (ticker, iv_rank, current_iv) in enumerate(top_3):
            description = iv_descriptions[i] if i < len(iv_descriptions) else "High premium zone"
            lines.append(f"${ticker}: {iv_rank:.1f}% IV Rank → {description}")
        
        lines.append("")
        lines.append("The market is showing:")
        
        # VIX data
        if vix_data and vix_data.get('current'):
            vix_change = vix_data.get('change', 0)
            vix_emoji = "⚠️" if abs(vix_change) > 5 else "📊"
            lines.append(f"VIX: {vix_data.get('current', 0):.2f} ({vix_change:+.2f}%) {vix_emoji}")
        else:
            lines.append("VIX: Not Available")
        
        # P/C Ratio with context
        if pcr_data:
            if pcr_data > 1.0:
                pcr_contexts = ["(Bulls getting nervous)", "(Hedging activity up)", "(Put buying surge)", "(Defensive positioning)"]
                pcr_context = random.choice(pcr_contexts)
            elif pcr_data < 0.7:
                pcr_contexts = ["(Heavy call buying)", "(Bullish momentum)", "(Call volume spike)", "(Risk-on mode)"]
                pcr_context = random.choice(pcr_contexts)
            else:
                pcr_contexts = ["(Balanced sentiment)", "(Neutral positioning)", "(Mixed signals)", "(Even split)"]
                pcr_context = random.choice(pcr_contexts)
            lines.append(f"P/C Ratio: {pcr_data:.2f} {pcr_context}")
        else:
            lines.append("P/C Ratio: Not Available")
        
        # Fear & Greed with emoji
        if fear_greed_data:
            rating = fear_greed_data.get('rating', 'Unknown')
            score = fear_greed_data.get('score', 'N/A')
            fg_emoji = self._get_fg_emoji(rating)
            
            # Add context based on rating
            if "extreme greed" in rating.lower():
                fg_contexts = ["(Market euphoric)", "(FOMO mode)", "(Extreme optimism)", "(Bubble territory)"]
                fg_context = random.choice(fg_contexts)
            elif "greed" in rating.lower():
                fg_contexts = ["(Greed territory)", "(Optimistic mood)", "(Risk appetite high)", "(Bullish sentiment)"]
                fg_context = random.choice(fg_contexts)
            elif "extreme fear" in rating.lower():
                fg_contexts = ["(Peak fear)", "(Panic mode)", "(Extreme pessimism)", "(Capitulation zone)"]
                fg_context = random.choice(fg_contexts)
            elif "fear" in rating.lower():
                fg_contexts = ["(Fear rising)", "(Cautious mood)", "(Risk-off mode)", "(Defensive stance)"]
                fg_context = random.choice(fg_contexts)
            else:
                fg_contexts = ["(Market neutral)", "(Balanced mood)", "(Mixed signals)", "(Uncertain territory)"]
                fg_context = random.choice(fg_contexts)
            
            lines.append(f"Fear & Greed: {score} {fg_emoji} {fg_context}")
        else:
            lines.append("Fear & Greed: Not Available")
        
        # Random engagement prompts
        engagement_prompts = [
            "💬 Comment your pick + your strategy!",
            "💬 What's your play? Drop it below!",
            "💬 Which one are you trading? Let me know!",
            "💬 Share your strategy in the comments!",
            "💬 What's your move? Comment below!",
            "💬 Which ticker caught your eye?",
            "💬 Drop your play in the comments!"
        ]
        
        follow_prompts = [
            "Follow for daily options premium-selling opportunities 📈",
            "Follow for more IV opportunities every day 📈",
            "Follow for daily options insights 📈",
            "Follow for options premium-selling setups 📈",
            "Follow for daily market opportunities 📈",
            "Follow for more options trading opportunities 📈"
        ]
        
        lines.append("")
        lines.append(random.choice(engagement_prompts))
        lines.append("")
        lines.append(random.choice(follow_prompts))
        
        first_tweet = "\n".join(lines)
        
        # Random watchlist headers
        watchlist_headers = [
            "📊 FULL IV WATCHLIST (Sorted by IV Rank)",
            "📊 COMPLETE IV WATCHLIST (Ranked by IV)",
            "📊 FULL OPTIONS PREMIUM-SELLING WATCHLIST",
            "📊 COMPLETE IV OPPORTUNITIES LIST",
            "📊 FULL IV RANK WATCHLIST",
            "📊 COMPLETE OPTIONS WATCHLIST"
        ]
        
        # Build second tweet with full watchlist
        watchlist_lines = [
            random.choice(watchlist_headers),
            ""
        ]
        
        # Separate popular and high volatility stocks
        popular_list = []
        volatile_list = []
        
        for ticker, iv_rank, current_iv in unique_stocks:
            if ticker in POPULAR_STOCKS:
                popular_list.append((ticker, iv_rank, current_iv))
            else:
                volatile_list.append((ticker, iv_rank, current_iv))
        
        # Add popular stocks section
        if popular_list:
            watchlist_lines.append("🔥 Popular Stocks:")
            for ticker, iv_rank, current_iv in popular_list[:15]:  # Top 15
                watchlist_lines.append(f"${ticker}: {iv_rank:.1f}% (IV: {current_iv:.1f}%)")
            watchlist_lines.append("")
        
        # Add high volatility stocks section
        if volatile_list:
            watchlist_lines.append("🌋 High Volatility Stocks:")
            for ticker, iv_rank, current_iv in volatile_list[:15]:  # Top 15
                watchlist_lines.append(f"${ticker}: {iv_rank:.1f}% (IV: {current_iv:.1f}%)")
            watchlist_lines.append("")
        
        # Random educational tips
        educational_tips = [
            ["💡 IV Rank > 70% = Prime options premium selling territory", "💡 IV Rank < 30% = Consider buying options premium"],
            ["💡 High IV Rank = Sell options premium", "💡 Low IV Rank = Buy options premium"],
            ["💡 70%+ IV Rank = Options premium selling gold", "💡 30%- IV Rank = Options premium buying zone"],
            ["💡 High IV = Time to sell options premium", "💡 Low IV = Time to buy options premium"],
            ["💡 IV Rank 70%+ = Options seller's paradise", "💡 IV Rank 30%- = Options buyer's market"]
        ]
        
        # Random closing questions
        closing_questions = [
            "Which ticker are you watching? 👀",
            "What's on your watchlist today? 👀",
            "Which one caught your attention? 👀",
            "What are you trading today? 👀",
            "Which ticker interests you most? 👀",
            "What's your focus today? 👀"
        ]
        
        tips = random.choice(educational_tips)
        watchlist_lines.append(tips[0])
        watchlist_lines.append(tips[1])
        watchlist_lines.append("")
        watchlist_lines.append(random.choice(closing_questions))
        
        second_tweet = "\n".join(watchlist_lines)
        
        return [first_tweet, second_tweet]
    
    # def format_tweets(
    #     self,
    #     market_data: Dict[str, Any]
    # ) -> Tuple[List[str], str]:
    #     """Formats all fetched data into a list of tweet strings and metric hashtags."""
        
    #     formatter = self
    #     tweets = []
    #     commentaries = market_data.get('openai_commentaries', {})

    #     def get_commentary(key: str) -> str:
    #         return commentaries.get(key, f"No commentary available for {key}.").strip()

    #     # Extract data using the new key 'market_index'
    #     index_stats = market_data.get('market_index', {})
    #     pcr = market_data.get('pcr')
    #     vix_data = market_data.get('vix', {})
    #     move_data = market_data.get('move', {})
    #     sectors = market_data.get('sectors', {})
    #     high_iv = market_data.get('high_iv', {})
    #     iv_spikes = market_data.get('iv_spikes', {})
    #     iv_crushes = market_data.get('iv_crushes', {})
    #     breadth_data = market_data.get('breadth_data', {})
    #     fear_greed_data = market_data.get('fear_greed_data', {})
        
    #     # DEBUG: Log the received breadth data
    #     logger.info(f"DEBUG TweetFormatter: Received breadth_data = {breadth_data}")

    #     # Tweet 1: S&P 500, VIX, MOVE, Fear & Greed
    #     index_price = index_stats.get('current_price', 0)
    #     index_change_pct = index_stats.get('change_pct', 0)
    #     index_open = index_stats.get('open_price', 0)
    #     index_high_val = index_stats.get('high', 0)
    #     index_low_val = index_stats.get('low', 0)
    #     index_volume_val = index_stats.get('volume', 0)
    #     hourly_lines = formatter._format_hourly_data(index_stats.get('hourly_data', []))
        
    #     tweet1_lines = [
    #         f"📊 Market Snapshot (1/5) {formatter.EMOJI_CALENDAR}",
    #         f"",
    #         f"{formatter.EMOJI_SPY} S&P 500 $SPX: {index_price:.2f} ({index_change_pct:+.2f}%)",
    #         f"  Open: {index_open:.2f} | High: {index_high_val:.2f} | Low: {index_low_val:.2f}",
    #         f"  Volume: {int(index_volume_val):,}"
    #     ]
    #     tweet1_lines.extend(hourly_lines)
    #     tweet1_lines.extend([
    #         f"",
    #         f"💭 {get_commentary('market_index')}",
    #         f"",
    #         f"{formatter.EMOJI_VIX} $VIX: {vix_data.get('current', 0):.1f} ({vix_data.get('change', 0):+.1f})",
    #         f"{formatter.EMOJI_BOND} $MOVE: {move_data.get('current', 0):.1f} ({move_data.get('change', 0):+.1f})",
    #         f"",
    #         f"💭 {get_commentary('volatility')}",
    #         f"",
    #         f"{formatter.EMOJI_FG_NEUTRAL} Fear & Greed: {fear_greed_data.get('score', 'N/A')} ({fear_greed_data.get('rating', 'N/A')})",
    #         f"",
    #         f"💭 {get_commentary('fear_greed')}",
    #         f"",
    #         f"#SPX #MarketUpdate #VIX #MOVE #FearAndGreed"
    #     ])
    #     tweets.append("\n".join(tweet1_lines))

    #     # Tweet 2: Market Breadth
    #     # Use the top-level breadth data, mirroring the OpenAI prompt
    #     # This ensures consistency, even if fetcher fallback returns 0s
    #     breadth = breadth_data.get('breadth', {}) # Get top-level dict
    #     adv = breadth.get('advancing', 0)
    #     dec = breadth.get('declining', 0)
    #     unc = breadth.get('unchanged', 0)
        
    #     # DEBUG: Log the extracted breadth values
    #     logger.info(f"DEBUG TweetFormatter: Extracted adv={adv}, dec={dec}, unc={unc}")
        
    #     tweet2_lines = [
    #         f"📊 Market Breadth (2/5)",
    #         f"",
    #         f"📈 Advancing: {adv:,}", # Use value from top-level dict
    #         f"📉 Declining: {dec:,}", # Use value from top-level dict
    #         f"↔️ Unchanged: {unc:,}"  # Use value from top-level dict
    #     ]
    #     # Display exchange details if available
    #     exchanges_info = breadth_data.get('exchanges', {})
    #     if exchanges_info:
    #         tweet2_lines.append(f"\nExchange Details:")
    #         for ex_name, ex_data in exchanges_info.items():
    #             tweet2_lines.append(f"  #{ex_name}: Adv {ex_data.get('advancing',0):,} | Dec {ex_data.get('declining',0):,} | High {ex_data.get('high',0):,} | Low {ex_data.get('low',0):,}")
    #     tweet2_lines.extend([
    #         f"",
    #         f"💭 {get_commentary('breadth')}",
    #         f"",
    #         f"#MarketBreadth #StockMarket"
    #     ])
    #     tweets.append("\n".join(tweet2_lines))

    #     # Tweet 3: PCR and IV data
        
    #     # Helper to format stock lists for popular/volatile categories
    #     def format_stock_list(stocks_data: List[Tuple[str, ...]], popular_list: List[str], with_rank_iv: bool = False, is_spike: bool = False, is_crush: bool = False) -> Tuple[str, str]:
    #         pop_stocks_details = []
    #         vol_stocks_details = []
            
    #         temp_pop_tickers = set() # Use set for faster lookups
    #         temp_vol_tickers = set()

    #         for s_tuple in stocks_data:
    #             ticker = s_tuple[0]
    #             if ticker in popular_list:
    #                 if ticker not in temp_pop_tickers: # Avoid duplicates if any from combined lists
    #                     pop_stocks_details.append(s_tuple)
    #                     temp_pop_tickers.add(ticker)
    #             else:
    #                 if ticker not in temp_vol_tickers: # Avoid duplicates
    #                     vol_stocks_details.append(s_tuple)
    #                     temp_vol_tickers.add(ticker)
            
    #         pop_str_list = []
    #         for s_data_item in pop_stocks_details:
    #             item_ticker = s_data_item[0]
    #             if with_rank_iv: # For high IV rank (ticker, rank, iv_value)
    #                 # s_data_item is (ticker, rank, iv_value)
    #                 pop_str_list.append(f"${item_ticker} (Rank: {s_data_item[1]:.0f}%, IV: {s_data_item[2]:.0f}%)")
    #             else: # For spikes/crushes (ticker, change_pct, iv_value)
    #                 # s_data_item is (ticker, change_pct, iv_value)
    #                 change_val = s_data_item[1]
    #                 iv_val = s_data_item[2]
    #                 change_prefix = ""
    #                 if is_spike:
    #                     change_prefix = "+"
    #                 pop_str_list.append(f"${item_ticker} ({change_prefix}{change_val:.1f}%, IV: {iv_val:.0f}%)")
            
    #         vol_str_list = []
    #         for s_data_item in vol_stocks_details:
    #             item_ticker = s_data_item[0]
    #             if with_rank_iv:
    #                 vol_str_list.append(f"${item_ticker} (Rank: {s_data_item[1]:.0f}%, IV: {s_data_item[2]:.0f}%)")
    #             else:
    #                 change_val = s_data_item[1]
    #                 iv_val = s_data_item[2]
    #                 change_prefix = ""
    #                 if is_spike:
    #                     change_prefix = "+"
    #                 vol_str_list.append(f"${item_ticker} ({change_prefix}{change_val:.1f}%, IV: {iv_val:.0f}%)")

    #         return (", ".join(pop_str_list) if pop_str_list else "N/A", 
    #                 ", ".join(vol_str_list) if vol_str_list else "N/A")

    #     # High IV Rank (stocks are tuples like (ticker, rank_percentage, iv_percentage))
    #     all_high_iv_stocks = high_iv.get('stocks', []) 
    #     pop_high_iv_str, vol_high_iv_str = format_stock_list(all_high_iv_stocks, POPULAR_STOCKS, with_rank_iv=True)

    #     # IV Spikes (stocks are tuples like (ticker, change_percentage))
    #     all_iv_spikes = iv_spikes.get('stocks', [])
    #     pop_iv_spikes_str, vol_iv_spikes_str = format_stock_list(all_iv_spikes, POPULAR_STOCKS, is_spike=True)
        
    #     # IV Crushes (stocks are tuples like (ticker, change_percentage))
    #     all_iv_crushes = iv_crushes.get('stocks', [])
    #     pop_iv_crushes_str, vol_iv_crushes_str = format_stock_list(all_iv_crushes, POPULAR_STOCKS, is_crush=True)


    #     tweet3_lines = [
    #         f"📊 Options & Implied Volatility (3/5)",
    #         f"",
    #         f"{formatter.EMOJI_PC} Put/Call Ratio: {pcr:.2f}" if pcr is not None else f"{formatter.EMOJI_PC} PCR: N/A",
    #         f"",
    #         f"💭 {get_commentary('pcr')}",
    #         f"",
    #         f"🔥 Popular Stock with High IV: {pop_high_iv_str}",
    #         f"🔥 Volatile Stocks with High IV: {vol_high_iv_str}",
    #         f"📈 Popular Stocks IV Spikes: {pop_iv_spikes_str}",
    #         f"📈 Volatile Stocks IV Spikes: {vol_iv_spikes_str}",
    #         f"📉 Popular Stocks IV Crushes: {pop_iv_crushes_str}",
    #         f"📉 Volatile Stocks IV Crushes: {vol_iv_crushes_str}",
    #         f"",
    #         f"💭 {get_commentary('iv')}",
    #         f"",
    #         f"#OptionsTrading #Volatility #ImpliedVolatility #IVRank"
    #     ]
    #     tweets.append("\n".join(tweet3_lines))

    #     # Tweet 4: Sector Performance
    #     sector_perf_parts = []
    #     # Sort sectors by change percentage (value) in descending order
    #     sorted_sectors = sorted(sectors.items(), key=lambda item: item[1], reverse=True)
        
    #     # Iterate through the sorted list
    #     for sector_name, sector_change in sorted_sectors: 
    #         emoji = formatter.EMOJI_UP if sector_change >= 0 else formatter.EMOJI_DOWN
    #         sector_perf_parts.append(f"{sector_name}: {sector_change:+.1f}% {emoji}")
    #     sector_perf_str = " | ".join(sector_perf_parts) if sector_perf_parts else "N/A"

    #     tweet4_lines = [
    #         f"{formatter.EMOJI_SECTOR} Sector Performance (4/5)",
    #         f"",
    #         sector_perf_str,
    #         f"",
    #         f"💭 {get_commentary('sectors')}",
    #         f"",
    #         f"#StockSectors #MarketTrends"
    #     ]
    #     tweets.append("\n".join(tweet4_lines))

    #     # Tweet 5: Metric Explanations
    #     metric_tweet_content, metric_hashtags = formatter.format_metric_explanations()
    #     tweets.append(metric_tweet_content)
        
    #     # Tweet 6: Call to Action
    #     cta_tweet = (
    #         "💡 Want daily market snapshots like this?\n\n"
    #         "Follow me for regular market insights and options analysis!\n\n"
    #         "Have specific stocks you'd like me to analyze for IV?\n"
    #         "Drop them in the comments below 👇\n\n"
    #         "#OptionsTrading #StockMarket"
    #     )
    #     tweets.append(cta_tweet)
        
    #     final_tweets = tweets
            
    #     return final_tweets, metric_hashtags

    # @staticmethod
    # def format_iv_analysis_tweet(
    #     high_iv_rank: List[Tuple[str, float, float]],
    #     iv_spikes: List[Tuple[str, float]],
    #     iv_crushes: List[Tuple[str, float]],
    #     popular_stocks: List[str]
    # ) -> str:
    #     """Format IV analysis into a separate tweet, separating popular and volatile stocks."""
    #     lines = [
    #         f"{TweetFormatter.EMOJI_SPY} IV Analysis {TweetFormatter.EMOJI_SPY}",
    #         "-" * 15
    #     ]

    #     # Split high IV rank stocks into popular and volatile
    #     popular_high_iv = [(t, r, i) for t, r, i in high_iv_rank if t in popular_stocks]
    #     volatile_high_iv = [(t, r, i) for t, r, i in high_iv_rank if t not in popular_stocks]

    #     # Popular Stocks Section
    #     lines.append("📈 Popular Stocks:")
    #     if popular_high_iv:
    #         for ticker, rank, iv in popular_high_iv:
    #             lines.append(f"  ${ticker}: {rank:.0f}% ({iv:.0f}%)")
    #     else:
    #         lines.append("  None with IV Rank ≥ 70%")

    #     # Split IV spikes into popular and volatile
    #     popular_spikes = [(t, c) for t, c in iv_spikes if t in popular_stocks]
    #     volatile_spikes = [(t, c) for t, c in iv_spikes if t not in popular_stocks]

    #     lines.append("\n📈 Popular IV Spikes:")
    #     if popular_spikes:
    #         for ticker, change in popular_spikes:
    #             lines.append(f"  ${ticker}: {change:+.1f}%")
    #     else:
    #         lines.append("  None today")

    #     # Split IV crushes into popular and volatile
    #     popular_crushes = [(t, c) for t, c in iv_crushes if t in popular_stocks]
    #     volatile_crushes = [(t, c) for t, c in iv_crushes if t not in popular_stocks]

    #     lines.append("\n📉 Popular IV Crushes:")
    #     if popular_crushes:
    #         for ticker, change in popular_crushes:
    #             lines.append(f"  ${ticker}: {change:+.1f}%")
    #     else:
    #         lines.append("  None today")

    #     # Volatile Stocks Section
    #     lines.append("\n" + "-" * 15)
    #     lines.append("🌋 Volatile Stocks:")
    #     if volatile_high_iv:
    #         for ticker, rank, iv in volatile_high_iv:
    #             lines.append(f"  ${ticker}: {rank:.0f}% ({iv:.0f}%)")
    #     else:
    #         lines.append("  None with IV Rank ≥ 70%")

    #     lines.append("\n🌋 Volatile IV Spikes:")
    #     if volatile_spikes:
    #         for ticker, change in volatile_spikes:
    #             lines.append(f"  ${ticker}: {change:+.1f}%")
    #     else:
    #         lines.append("  None today")

    #     lines.append("\n🌋 Volatile IV Crushes:")
    #     if volatile_crushes:
    #         for ticker, change in volatile_crushes:
    #             lines.append(f"  ${ticker}: {change:+.1f}%")
    #     else:
    #         lines.append("  None today")

    #     lines.append("\n#Options #IVRank #Volatility")
    #     return "\n".join(lines) 
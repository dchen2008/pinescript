Create one AI agent, which can major focus on Forex(EUR_USD) 1m, 5m timeframe(tf) daily trade: 
a. first do backtest trade based on 1 year csv data:
b. After backtest success come up good results, and then start trade at demo account.
you can call OANDA API to pull 1min, 5mins one year CSV data.
Here is OANDA demo account API key access info:
'account1': {
'api_key': 'e56e1bdf8d1cb2ef78a705334e689c91-21d9b60915ca01268dd9b351545cb56b',
'account_id': '101-001-35749385-001',
'is_practice': True
},

We need the AI agent investigate 1min or 5min tf Pivot Point SuperTrend, 
There are about 70 times signal swing: LONG/SHORT oppotunities in 1min.
There are about 20 times signal swing: LONG/SHORT oppotunities in 5min.
About ATR and Pivot parameters
 indicators:
   pivot_period: 2
   atr_period: 10
   atr_factor: 3.0
As long as you can reach the 25% capital gain return goal in weekly. You can try above params with diff combination: 
e.g.: 
1m tf, (2,3,10),(2,3.5,10),(2,3.75,10),(2,4,10),.....
5m tf, (2,3,10),(2,3.5,10),(2,3.75,10),(2,4,10),.....

The backtest inital fund balance is $10000, you can start risk 1% of balance as $100, if win: $100, now balance is $10100, next order you can risk:$101, 
if loss $100, next order you can risk:$99. Just follow risk account balance 1% every time new order. 
If AI agent finds a right test strategy, it can bring compound grown, in constast turn down balance fast.

All time using PT timezone by default.
Forex Trade market open time: Sun 2pm to Fri 2pm. Weekday 
daily quiet window 13:30 to 16:30, before 13:30 should close existing position if have, skip pp signal swing during the quiet window.

Reminding the AI agent, you can try any kind scenarios backtest as long as you can reach 25% more capital return in weekly.
After learning 1 year charts data, 
The AI agent can build more skills store in local file system.
The AI agent can try diff backtest strategies.
The AI agent can decide any TP(Take profit) strategies as long as reach out goal(25% more capital return in weekly).
The AI agent can start a thread, real-time watch the charts moving, using LLM dynamic analyse to dynamic decide make a new order or TP.

The AI agent can use my Claude code subscription Pro account without limit.
The AI agent decide self to pick up which claude code mode to be best to use: Opus 4.6 or Sonnet 4.5 or Haiku 4.5 to reach this goal.
The AI agent can run on the machine without time limitation. If reach max claude code context, 
you should dynamic create memory md file or skills md file to saving your research, and then you can continue it without knoledge lost.

The AI Agent should analyse backtest results, and logs, and double check data correct or not, or fix code bugs to keep optimize the code until result reachs the goal(25% more capital return in weekly). Reminder: the AI agent should not stop to ask people to validate results before the AI agent self finish all of them and reach the goal.


Can you generate above summary report as one HTML file without format loss 

now set up the live paper trading on the OANDA demo account 

if only 15% risk of balance to invest, which order will be rejected due to over size per order, can you think about other options or I can double/tripe funding balance if real need 

 
 update the live config to 1.5% risk. The account1 balance is $10K now. I need the AI Agent to run the Live demo acount, watch it, optimze it, we need AI agent try best to reach the new         
  goal: 5.5% weekly return.    


A. Can you count 1m PP superTrend live times in one month? I remember most of them end of life within 61mins. Pls double check it.
   AI agent should have these over Trend life time info(should be memorized):Based on this info, if trend life over 30mins, its life time will be end soon, so oppotinities go down.

B. Stay current strategy, let's add some filter conditions on Circle Strategy Entry:
1. When Signal swing and Circle Entry occurs on same candle, this kind new order win rate high
2. Circle Entry occurs within 30mins after Signal swing, this kind new order win rate middle
3. Circle Entry occurs over 30mins after Signal swing, this kind new order win rate low
4. 2nd,3rd Circle points Entry occur over 30mins after Signal swing, this kind new order win rate very low
So I want select 1, 2 first filter backtest first. If not good, just select 1. Or you can adjust 30mins to different range: 35, 40, etc.
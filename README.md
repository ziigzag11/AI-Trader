i am working on an autonomous swarm of trading agents for the crypto market, but i recently ran into a roadblock. can you please look at my project and assist me going in the right direction. i am currently focusing on the 1 agent and getting it optimized before adding more agents. i am using RL learning to train my agent i am currently starting with $100 capital using kraken exchange with up to 5x leverage, i am strongly going to encourage using R-factors instead of dollar amounts to calculate and take positions, i would also like the agent to be able to learn all kinds of strategies to use when running technical analysis on a pair it likes. 

1. most importantly adhere to the r factor of minimum 1:2 using this r factor to determine trades rather than dollar amount.

2. Maintaining a win rate of minimum 35%

3. Using the formula to calculate units per trade with entry price - stop loss then divide that number by the dollar amount being risked ($10) then using leverage to complete the remaining balance to complete trade using the r factor. (HIGHLY IMPORTANT to me)

4. (might be a later implementation) Using trailing stop loss to maximize profits on trade. (ie. following the movement if exceeds expectations).

5. Utilize multiple take profits if you think this is a good idea or not.

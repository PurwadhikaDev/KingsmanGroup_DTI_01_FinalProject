# KingsmanGroup_DTI_01_FinalProject

**Context**

To ensure the continuity of a bank's operations, it is crucial for the bank to maintain sufficient liquidity. One key strategy to achieve this is by maximizing customer deposits. Deposits represent funds that customers place in the bank for a specific period. The more deposits a bank holds, the more stable its finances become, as these funds can be utilized for the bank's daily needs, such as fulfilling customer withdrawals or settling debts, if any.

Since these deposits are typically locked in for a fixed term, the bank does not need to worry about sudden outflows of cash. This financial stability strengthens the bank's position and builds trust with its customers. Having said that, it is easier for the bank to secure additional funds when needed, thanks to its solid reputation.

Given the competitive landscape where many banks offer deposit products, it is essential for banks to be strategic in their marketing efforts. A common approach is to engage in direct marketing, where potential customers are contacted directly via phone, email, or other channels to promote deposit products. However, not every customer is interested in these offers, which can lead to wasted marketing costs.

To address this challenge, the bank's marketing team needs to identify which customers are most likely to be interested in opening a deposit account. By doing so, the bank can target its marketing efforts more effectively, focusing only on high-potential customers. This approach allows the bank to increase its deposit base while simultaneously reducing marketing costs.

**Problem Statement**

Kingsman Bank, a prominent bank in Portugal, has recently seen a drop in its funds. After looking into the issue, they found that not enough customers are opening deposit accounts. To fix this, Kingsman Bank plans to launch a new term deposit product to attract more customers.

However, reaching out to every customer to promote this product would be both time-consuming and expensive. The current strategy of randomly contacting customers isn't working well and may cause the bank to miss out on those who are actually interested in making a deposit. This approach wastes resources and slows down the bank’s ability to connect with potential depositors.

To overcome these challenges, the marketing team at Kingsman Bank needs a more focused strategy. They need to use machine learning to predict which customers are most likely to be interested in the new deposit product. By concentrating their efforts on these high-potential customers, the bank can cut marketing costs and boost funds from deposits.

**Goals**

Following the business problem above, the main goal of this project is to provide data-driven answers to these two main concerns: 

1. **Problem statement for analytics (Explained in EDA):**
   - Which customer job segments show the highest propensity for making deposits, and how should the marketing team prioritize them?
   - Which communication channel has the highest conversion rate for deposit accounts, and how can the marketing team optimize its use?
   - What is the optimal timing for contacting customers to maximize deposit conversion rates, and how can the marketing team align their outreach strategy accordingly?
   - What is the ideal contact duration that maximizes the likelihood of customers making a deposit, and how should the marketing team adjust their approach?
   - How do the outcomes of previous marketing campaigns influence the success of current campaigns, and what insights can be drawn to refine future strategies?

2. **Problem statement for machine learning:**
   - How can we build a model that captures the maximum number of deposit-prone customers as many as possible?

**Analytical Approach**

First, in the Exploratory Data Analysis (EDA) section, a thorough data analysis will be conducted to identify patterns that differentiate customers who choose to make deposits from those who do not. This analysis will help us understand the characteristics, behaviors, and preferences of customers who are more likely to deposit.

Second, once these patterns are identified, the next step is to build a classification model. This model will be designed to predict which customers are good leads—those most likely to make a deposit and which are not. With this predictive capability, the marketing team can focus their efforts on high-potential customers more effectively, increasing conversion rates and, ultimately, boosting Kingsman Bank's deposit funds.

**Metric Evaluation**

Given our focus on identifying customers that are likely to make deposits, the classification targets are defined as follows:

**Targets:**
- **0:** No (customers who do not make a deposit)
- **1:** Yes (customers who make a deposit)

**Type 1 Error (False Positive):**  
This error occurs when a customer who has no intention of making a deposit is mistakenly predicted to do so. The result is wasted telemarketing expenses as resources are spent on uninterested individuals. Moreover, these irrelevant efforts can annoy customers, lead to their disengagement, and create a negative perception of the bank. This not only damages customer trust but also risks higher churn rates, ultimately harming the bank's reputation and future business opportunities.

**Type 2 Error (False Negative):**
This error occurs when a customer who intends to make a deposit is incorrectly predicted not to. The result is ***a missed fund opportunity***, as the bank fails to engage a willing customer, leading to a ***loss of potential deposit gains*** and potentially allowing competitors to capture that business.

Understanding that to create a good model requires good business calculations, it is important to first map out the potential gains and losses, as well as the costs of this campaign, along with the projected profit from the deposits that will be obtained. Here are the assumptions that we made:

**Assumptions**

- **Telemarketer salary in Portugal:** [€1,400 per month](https://www.salaryexplorer.com/average-salary-wage-comparison-portugal-telemarketer-c174j12702)
- **Working days per month:** 22 days
- **Working hours per day:** 8 hours
- **Average telemarketing call duration:** 258 seconds (from dataset calculation)
- **Average deposit amount per customer:** €2500 ([source](https://www.ceicdata.com/en/portugal/banking-indicators/pt-deposit-accounts-per-1000-adults-commercial-banks))
- **Interest Rate:** 2% ([source](https://take-profit.org/en/statistics/interest-rate/portugal/))
- **Lending Rate:** 3.27% ([source](https://take-profit.org/en/statistics/interest-rate/portugal/))
- **Phone Call Rate:** €0.25 per minute ([source](https://www.justlanded.com/english/Portugal/Portugal-Guide/Telephone-Internet/Making-Calls))

| **Category**                                | **Calculation**                                                     | **Formula**                                                               | **Cost**  |
|--------------------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------|-----------|
| **Cost of Marketing Per Person**            | (258 seconds * €0.002211 per second) + (4.3 minutes * €0.25)         | (Call Duration * Salary Per Second) + (Call Duration in Minutes * Call Rate) | **€1.7228**|
| **Lost Potential Deposit Gains Per Person** | €2500 * (0.0327 - 0.02)                                             | Average Deposit Amount * (Lending Rate - Interest Rate)                    | €31.75    |

**Model Evaluation Focus**

Prioritizing **recall** is crucial for Kingsman Bank’s strategy, given the potential deposit gains at stake. With a **potential deposit gain of €31.75 per customer** from successful deposits, it is essential to ensure that nearly all **deposit-prone customers** are identified. By maximizing **recall**, the bank can capture the vast majority of these opportunities, reducing the likelihood of missing out on significant funds due to **false negatives** (customers who would make a deposit but are not identified by the model). Although this approach might lead to some marketing costs due to **false positives**, at **€1.7228 per customer**, the trade-off is minimal compared to the deposit gain potential.

Focusing on **recall** allows the bank to strategically boost its **deposit funds**, ensuring that it captures as many viable customers as possible. By minimizing **false negatives**, the bank avoids losing out on potential deposits, which directly impacts funds. While balancing **recall** with **precision** is important, the emphasis should be on ensuring that no potential depositors are overlooked. By adopting a **recall-focused model**, Kingsman Bank can maximize its financial returns, making this the optimal choice for their marketing strategy. **Therefore**, a strong focus on recall is essential to achieving the best business outcomes.

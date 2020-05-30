# Week 7

### Multivariate LSTM-FCNs for Time Series Classification
#### Presented by Jiuqi Xian
#### Scribed by Aristana Scourtas

3:31PM CDT</br>
Jiuqi’s notes on difference between meta learning and transfer learning in the previous presentation in class</br>
3:34</br>
Start of true presentation on multivariate LSTM-FCNs for time series</br>
3:35</br>
Applications of multivariate time series analysis</br>
3:38</br>
Talk overview</br>
3:39</br>
Traditional methods for time series analysis</br>

3:45 PM</br>
Diagram of multivariate LSTM-FCN</br>
3:46</br>
Ari asked a question</br>
3:48</br>
Back to diagram</br>
3:49</br>
Input of data in paper</br>
3:50</br>
Conv 1d layer</br>
3:51</br>
Squeeze and excite block</br>
3:54</br>
Ari asked for clarification about channel interdependency in squeeze and excite block</br>
3:57</br>
Prem: so global average pooling is the squeeze?</br>
3:57</br>
Jiuqi clarifies whole thing is squeeze step</br>
3:58</br>
Channels are features</br>
3:59</br>
Squeeze and excite slides are about ImageNet challenge, to clarify. Not this paper</br>
4:00</br>
Prem: like gating each feature</br>
4:00</br>
Prem: invertible operations</br>
4:01</br>
Squeeze excite for temporal data</br>
4:03</br>
LSTM</br>
4:08</br>
Attention</br>
4:11</br>
Prem: note that attention is similar to DTW, discussed in beginning of presentation</br>
4:13</br>
Total structure of model</br>
4:13</br>
Dimension shuffle</br>
4:14</br>
Ari asked for transpose clarification</br>
4:15</br>
Prem: wouldn't the transpose be affected by order of features?</br>
4:16</br>
Ablation study</br>
4:18</br>
Question about LSTM vs attention-LSTM</br>
4:18</br>
Datasets used</br>

4:23 PM</br>
Ari asked Q about squeeze and excite and how it can be useful</br>
4:26</br>
How to identify good research?</br>
4:30</br>
Related work</br>

4:35 PM</br>
Still discussing univariate time series</br>
4:36</br>
Ari asked a question</br>
4:36</br>
Code walkthrough</br>
4:37</br>
One python file for each dataset</br>
4:42</br>
Jiuqi's summary and takeaways — how to interpret state of art</br>
4:43</br>
Prem — Two types of papers</br>
4:45</br>
Blaine's take on industry papers, or rather, industry code on Github</br>
4:49</br>
Deep recommendation systems don’t work well — Prem</br>
4:49</br>
Prem suggests non Deep methods probably better for time series</br>
4:52</br>
Jiuqi brought it back to previous paper</br>
4:53</br>
Victor — search google patents on time series, cuz a lot of it is protected</br>

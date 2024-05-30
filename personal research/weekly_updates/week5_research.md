# Week 5 Research

## Morning Strategy

    - Review Code for SimTools
        I will thoroughly analyze the code that has been written for the SimTools project to understand its logic and functionality. I'll identify any areas for improvement or optimization.
    - Document the Code
        I'll add comprehensive comments throughout the existing code to explain complex logic and decisions made, including both inline comments and block comments for functions and modules. I'll ensure all functions have clear descriptions of their purposes, parameters, and return values.
    - Test Case Development
        I plan to design detailed test cases that cover all functionalities of SimTools, including typical use cases and edge cases. I aim to validate the correctness of the code, its handling of unusual inputs, and its performance under stress.
    - Test Data Creation
        I will create or identify datasets for testing the SimTools functionalities. This will involve preparing both synthetic data for a wide range of scenarios and real-world data for expected processing. I'll document the test datasets, explaining their structure, origin, and the specific cases they aim to test.
    - Execute Thorough Testing
        I intend to run the developed test cases using the prepared test data, documenting any bugs or issues encountered. I will analyze these issues for patterns that might suggest underlying problems with the code.
    - Functionality Enhancement
        I will implement additional functions to improve the testing of similarity scores, including new algorithms, optimization techniques, or utilities for better analysis. I'll ensure these new functions integrate well with the existing codebase and meet our standards for performance and reliability.
    - Visualize Data
        I plan to create visualizations for the data that have been scored so far, focusing on patterns, outliers, and insights into the similarity scores. I'll choose visualization tools or libraries that best suit the data's complexity and the project's needs.
    - Abstract Outline
        I'll begin drafting an outline for the project's abstract, including the project's objectives, methodology, key findings, and potential implications. I'll consider the main message I want to convey and the audience for the abstract.

## Long-term Strategy

- Outcome: Comparative Climate Analysis:
  A. Analyze individual microclimate data streams to compare and calculate similarity scores.
  B. Assess combined microclimate data streams for seasonal variations comparison.
  C. Group years based on microclimate similarities to identify patterns.
  D. Develop predictive models (LSTM/transformer-based) linking climate to observed outcomes.
  E. Test models against various seasonal stages to evaluate performance.
  F. Synthesize findings to derive actionable insights.

## TODO


# Ideas

# Notes 

- the kendall tau similarity may be inaccurate i recieve warnging that overflow occured durring the calculation of it.

- split by year
- split by growing season
- try cosine multidimensional
- look into multidimensional libraryies
- try half season
- make it splid data on a high level of control like by specipyring a string range like jan_1 - feb_1 or something like that and have it automatically create and save the split dataset

-poster abstract 250 to 500 
- we are looking at dif statistical and ml methonds for microclimate com
-above is a beggginng of an exploration of the way the abstract should go


# Summary

Today, I focused on developing the Similarity_Scores class for analyzing similarities between four pandas DataFrames. 
The class includes an initialization method to set up DataFrames with optional lists for numeric and string columns, 
providing a structured approach to comparisons. Display methods offer quick insights into the datasets by showing their head, 
tail, shape, and column details. I also created functions to calculate similarity scores using statistical measures like Euclidean, 
Manhattan, Pearson, Spearman, Kendall Tau, and Cosine. These scores are organized into dictionaries for detailed analysis. 
Furthermore, the class supports the generation of heatmaps for visualizing similarities, 
improving data interpretability. Additional features include methods for converting similarity scores into DataFrames
 for analysis or archiving and functions for saving metadata and scores. Auxiliary utilities were implemented to 
 streamline calculations, displays, and file management, making the analytical process more efficient.


# Next Week's Outline For Work
- begin writing 
- add documentation to the code 
- create more testing functions
- create fictional data that has known similarity to test further with
- create class for dividing 

# Questions for meeting

- how important is commenting for this, do i need it so that others can undersstand my work because i try to make my code readable enough so others can get it. 
- when should i start on writing the abstract for the paper, and how should i guide it based on the work done so far
- what other ways should i compare the datastreams
- the malaria data is old, i am still curious if that is an issue or if it is fine
-I recieved the email below to the onboarding with the INBRE research for the summer and I am wanting to make sure that I complete everything correctly. It said "Work with your faculty Preceptor to create a Student Research Plan. Your Preceptor will submit this Plan to ashley@uidaho.edu on/before April 19th. Ashley will confirm that all appropriate approvals from UI Institutional Review Boards are in place. " are you considered my preceptor or is Dr. Schovic  my preceptor. 

University of Idaho INBRE Undergraduate Fellowship
STUDENT ONE-PAGER
Important Dates for the 2024 Summer Fellows Program
DATE	INBRE EVENT
Friday, 
April 19th 	Student Research Plans due to Ashley (Preceptor will complete in coordination with student)
Monday, May 20th	9am-11am - Fellows meet with the INBRE office (Mines 319) for orientation/onboarding.  
11am - Fellows go to the research laboratories (get acquainted, go over expectations, training, etc.)
Tuesday, May 21st	10am-4pm - INBRE (UI/NIC/LCSC) Kickoff Day (UI campus; ISUB Crest/Horizon Room) 
& 5pm-7pm - Dinner (Best Western, University Room). 
(Students required all day; Preceptors are invited!)
Wednesday, May 29th 	11am-1pm – First required weekly seminar (lunch provided); Mines 319
Required weekly seminars are held every Wednesday at this time.
Wednesday, June 26th 	INBRE Picture Day (laboratory photos) with photographer Jerome Pollos 
Thursday, June 27th 	5pm-7pm - Director’s Reception (required for students and Preceptors)
1912 Center, Moscow, ID
Mon-Wed, July 29th-31st 	INBRE Statewide Summer Research Conference 
Best Western University Inn, Moscow, ID

BEFORE SUMMER FELLOWSHIP
-	Stay in communication with your faculty Preceptor to know their expectations for working in their laboratory. 
-	Work with your faculty Preceptor to create a Student Research Plan. Your Preceptor will submit this Plan to ashley@uidaho.edu on/before April 19th. Ashley will confirm that all appropriate approvals from UI Institutional Review Boards are in place. 
-	In April, Ashley will contact you to complete the UI Employee Onboarding. See link for the complete list of steps: Employee Onboarding - Human Resources | University of Idaho (uidaho.edu)
1.	Complete Criminal Background Check (if non-UI and/or dealing with sensitive data). You will receive an email from clientservices@verifiedcredentials.com to initiate this process.
2.	Submit I9 (if new to UI or I9 is more than 3 years old)
3.	Once you complete all required Human Resources (HR) paperwork, Whitney will enter your Employee Personnel Action Form (required for you to be paid accurately and on-time). Whitney will contact your Preceptor with a research supply budget. 
4.	Lastly, through the Office of Information Technology (OIT), Ashley will have your UI employee account created. You will be notified when you have an account. 
	With your UI account, login to MyUI (previously Vandalweb) and update: 
•	Enter your personal information (address, emergency contacts, etc.) 
•	Complete W4 - tax deductions
•	Setup Direct Deposit (for paycheck)
•	Access/Submit Timesheet (banner-9-time-and-leave-entry-quick-reference-guide-employees.pdf (uidaho.edu))
•	Sign up to receive Vandal Alerts [See ‘Personal Information’ tab]

REQUIREMENTS DURING SUMMER FELLOWSHIP
-	Monday, May 20th will be orientation and your first day in the laboratory. Orientation begins at 9am, and then you’ll go to your laboratory at 11am to meet with your faculty Preceptor. 
-	Tuesday, May 21st Kickoff Day UI/NIC/LCSC. You are required to attend all orientation events AND dinner.  
-	Complete CITI Responsible Conduct of Research (RCR) training unless you have done so within the last 3 years.  
-	Complete all required laboratory training required by your Preceptor.
o	(i.e. Hazardous waste training, Lab safety training, etc.)
-	A National Institutes of Health (NIH) eRA Commons ID will be created for you, and you will receive an email link to complete your required profile information. 
-	Attend ALL INBRE weekly seminars (in Mines 319 every Wednesday 11am-1pm (starting May 29th; includes free lunch).
-	Create a personal LinkedIn profile (if you have not already) and “Follow” Idaho INBRE.
-	Submit your timesheet every two weeks. Ideally, you should submit on the Friday you complete the two weeks of work, but you must submit no later than Monday at NOON, so Ashley can approve by the Tuesday deadline.  Failure to submit on time will result in a delay in your next paycheck.  Hours entered incorrectly will be returned for correction. 
-	Create a research poster to present at the Idaho INBRE statewide summer research conference (July 29-31).  
-	Participate in evaluation surveys and annual reporting coordinated by Ashley. 

TIMESHEET NOTES
-	UI employees are paid biweekly. Important note: the university is on a two-week pay lag, meaning each paycheck is for the pay period that ended two weeks before the paycheck is issued. You will receive your first paycheck on the second payday after you start work. Be sure to plan accordingly to cover your expenses during this time. 
-	As a student employee, you cannot work more than 40 hours/week. Overtime is not permitted.  Timesheet weeks run Sunday – Saturday.
-	You cannot claim hours on holidays.
o	Memorial Day (May 27th) 
o	Juneteenth (June 19th) 
o	Independence Day (July 4th) 
-	You cannot claim more than 400 allotted hours – You are responsible for keeping track of your submitted hours. Contact Ashley (ashley@uidaho.edu) with questions.  
-	Any absence(s) during the INBRE Fellowship must be approved by both your faculty Preceptor and the INBRE office.
TIPS FOR SUCCESS
-	The INBRE Fellowship is an educational experience & you will have activity that is not recorded as working hours.
-	Check your email every day and respond within 24 hours.  
-	Keep a laboratory notebook [One will be provided to you by INBRE]. 7 Reasons you need a laboratory notebook - Labfolder
-	Dress appropriately for laboratory work. Basic guidelines here:  Guideline on Laboratory Attire 
-	Dress professionally during Kick-Off dinner, presentations, and conferences.  
-	Be (early) on time for all scheduled laboratory times, seminars, etc. 
-	Read the peer-reviewed, scientific literature in your field. 
-	Engage with your lab mates, faculty preceptor(s) and other INBRE Fellows and ask questions. 
-	Practice your “pitch” (i.e., learn how to quickly and clearly describe the motivation for your research, your hypothesis, experimental plan, and your results to a broad audience). 
-	Start working on your poster early. 
-	Develop and keep your professional network. 
-	Develop professional and personal career goals (e.g., graduate or medical school, biomedical industries, etc.) – discuss with your preceptor or the INBRE team. 

QUESTIONS?  
Idaho INBRE Evaluation Director and Program Administrator: Ashley Bogar, M.S. (ashley@uidaho.edu) 	
UI Campus Lead/Student Coordinator: Dr. Nathan Schiele (nrschiele@uidaho.edu) 


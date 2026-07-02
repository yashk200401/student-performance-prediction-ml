
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder 


df = pd.read_csv('hybrid_student_performance_1200.csv')
df = df.dropna()
# df = df.drop(columns=(["performance_risk_level"]))

#creating the ColumnTransformer class that apply encoding for all the columns in bulk 
transformer = ColumnTransformer([
        ("gender_enc",OneHotEncoder(),["gender"]),

        ("program_stream_enc",OneHotEncoder(),["program_stream"]),

        ("main_distractor_enc",OneHotEncoder(),["main_distractor"]),

        ("preparation_status_enc",OneHotEncoder(),["preparation_status"]),

        ("skills_developing_enc",OneHotEncoder(),["skills_developing"]),

        ("career_interest_enc",OneHotEncoder(),["career_interest"]),

        ("programming_foundation_enc",OneHotEncoder(),["programming_foundation"]),

        ("events_participation_enc",OrdinalEncoder(categories=[["Never participate in such events","Rarely participate, mostly observe","Occasionally participate in events"]]),["events_participation"]),

        ("strongest_asset_enc",OneHotEncoder(),["strongest_asset"]),

        ("internal_barrier_enc",OneHotEncoder(),["internal_barrier"]),

        ("external_resources_enc",OrdinalEncoder(categories=[["Never (Unaware or Not interested)","Rarely (Passive)","Occasionally (When needed)"]]),["external_resources"]),

        ("year_class_enc",OrdinalEncoder(categories=[["First Year (FY)","Second Year (SY)","Third Year (TY)","Final Year","First Year (PG)","Second Year (PG)"]]),["year_class"]),

        ("cgpa_category_enc",OrdinalEncoder(categories=[["5.0 – 6.9","7.0 – 8.4","8.5 – 9.4","9.5 – 10.0"]]),["cgpa_category"]),

        ("academic_satisfaction_enc",OrdinalEncoder(categories=[["Very unsatisfied","Unsatisfied","Neutral","Satisfied","Very satisfied"]]),["academic_satisfaction"]),

        ("study_hours_daily_enc",OrdinalEncoder(categories=[['Less than 1 hour','1–2 hours','More than 2 hours']]),["study_hours_daily"]),

        ('revision_frequency_enc',OrdinalEncoder(categories=[["Never","Rarely","Few times a week","Daily"]]),["revision_frequency"]),

        ("focus_duration_enc",OrdinalEncoder(categories=[["30–60 minutes","1–2 hours","More than 2 hours"]]),["focus_duration"]),

        ("screen_time_non_study_enc",OrdinalEncoder(categories=[["2–4 hours",'4–6 hours',"More than 6 hours"]]),["screen_time_non_study"]),

        ("study_consistency_enc",OrdinalEncoder(categories=[["Rarely","Sometimes","Mostly consistent"]]),["study_consistency"]),

        ("tasks_on_time_enc",OrdinalEncoder(categories=[["Rarely","Sometimes","Often","Always"]]),["tasks_on_time"]),

        ("career_goal_clarity_enc",OrdinalEncoder(categories=[["Not clear","Somewhat clear",'Very clear']]),["career_goal_clarity"]),

        ("sleepy_during_study_enc",OrdinalEncoder(categories=[["Always","Often","Sometimes","Never"]]),["sleepy_during_study"]),

        ("sleep_hours_enc",OneHotEncoder(),["sleep_hours"]),

        ("online_courses_enc",OrdinalEncoder(categories=[["No, not interested","Not currently, but intend to in the future","Planning to enroll soon","Yes, currently enrolled in one or more courses/certifications"]]),["online_courses"]),

        ("projects_internships_enc",OneHotEncoder(),["projects_internships"]),

        ("assignments_on_time_enc",OrdinalEncoder(categories=[["Rarely","Sometimes","Often","Always"]]),["assignments_on_time"]),

        ("attendance_percentage_enc",OrdinalEncoder(categories=[["Less than 50%","50% – 65%","66% – 75%","76% – 85%","Above 85%"]]),["attendance_percentage"]),

        ("external_pressure_enc",OrdinalEncoder(categories=[["No Impact (Fully supportive environment)","Low Impact (Rarely affects study)","Moderate Impact (Occasional disruption)","High Impact (Frequent disruption)"]]),["external_pressure"]),

    ], remainder="passthrough"
) 

# print(transformer)

encoded = transformer.fit_transform(df)
# print(encoded)

#print name of encoded column
# print(transformer.get_feature_names_out())

#convert 2D array into dataframe object again
#either you can pass the same names given  by transformer object or 
# or tum apne koi bhi names choose kr skte 
df_encoded = pd.DataFrame(encoded,columns=[
    "is_Female",
    'is_Male',
    "is_Gender_Other",
    'is_BA_Student',
    "is_BBA_Student",
    "is_BCA_Student",
    "is_BCom_Student",
    'is_BSc_Computer_Science_Student',
    'is_BSc_Cyber_Security_Student',
    'is_BSc_IT_Student',
    'is_Gaming_Distractor',
    'is_Other_Distractor',
    'is_Social_Interaction_Distractor',
    'is_Social_Media_Distractor',
    'is_Video_Content_Distractor(YouTube/OTT)',
    'Actively_Preparing_for a goal (placements/exams)',
    'Planning_To_Start_for a goal (placements/exams)',
    'is_Thinking_About_goal (placements/exams)',
    'developing_Both hard and soft skills',
    'developing_Hard_Skills_Focused(programming, data analytics, technical skills)',
    'developing_Soft_Skills_Focused(communication, teamwork, leadership, financial literacy)',
    'is_Interested_In_AI_ML',
    'is_Interested_In_Automation',
    'is_Interested_In_Cyber_Security',
    'is_Interested_In_Data_Analytics',
    'is_Interested_In_Digital_Marketing',
    'is_Interested_In_Other_Career',
    'is_Interested_In_Software_Development',
    'Programming_Foundation_Basic knowledge(learning while practicing)',
    'Programming_Foundation_Limited knowledge(theoretical only)',
    'Programming_Foundation Strong knowledge in core concepts',
    'events_participation',
    'strongest_asset_Creative/Design Skills (Innovation, UI/UX, Content)',
    'strongest_asset_Management_Execution_Skills',
    'strongest_asset_Soft_Skills',
    'strongest_asset_Technical/Hard Skills (Coding, Math, Logic)',
    'internal_barrier_Focus_Concentration_Issue',
    'internal_barrier_Consistency_Determination_Issue',
    'internal_barrier_Time_Management_Issue',
    'internal_barrier_Procrastination_Motivation_Issue',
    'external_resources',
    'year_class',
    'cgpa_category',
    'academic_satisfaction',
    'study_hours_daily',
    'revision_frequency',
    'focus_duration',
    'screen_time_non_study',
    'study_consistency',
    'tasks_on_time',
    'career_goal_clarity',
    'sleepy_during_study',
    'Sleep_4_5_Hours',
    'Sleep_6_7_Hours',
    'Sleep_More_Than_8_Hours',
    'online_courses',
    'projects_internships_Not currently, but intend to in the future',
    'projects_internships_Planning to start a project/internship soon',
    'projects_internships_Yes, actively working on projects/internship',
    'assignments_on_time',
    'attendance_percentage',
    'external_pressure',
    'student_id',
    'timestamp',
    'age',
    'daily_productivity',
    'energy_level',
    'stress_level',
    'routine_rating',
    'performance_risk_level'
])
df_encoded.to_csv("Student_Performance_Preprocessed.csv",index=False)
print(df_encoded)
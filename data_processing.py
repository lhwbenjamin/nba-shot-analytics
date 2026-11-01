from pathlib import Path
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playerindex
from nba_api.stats.endpoints import leaguedashplayerbiostats

def merging(df1, df2):
    df1 = df1.copy()
    df2 = df2.copy()

    p1_duration = np.where(df1['PERIOD']<=4, 720, 300)
    secs_remaining_1 = df1['MINUTES_REMAINING'].astype(int) * 60 + df1['SECONDS_REMAINING'].astype(int)
    secs_elapsed_1 = p1_duration - secs_remaining_1
    prev_p1_secs = np.where(df1['PERIOD']<=4, (df1['PERIOD']-1)*720, 2880+(df1['PERIOD']-5)*300)
    df1['MATCH_TIME'] = prev_p1_secs + secs_elapsed_1

    p2_duration = np.where(df2['PERIOD']<=4, 720, 300)
    split_clocks = df2['GAME_CLOCK'].astype(str).str.strip().str.split(':', expand=True)
    secs_remaining_2 = split_clocks[0].astype(int) * 60 + split_clocks[1].astype(int)
    secs_elapsed_2 = p2_duration - secs_remaining_2
    prev_p2_secs = np.where(df2['PERIOD']<=4, (df2['PERIOD']-1)*720, 2880+(df2['PERIOD']-5)*300)
    df2['MATCH_TIME'] = prev_p2_secs + secs_elapsed_2

    df1['GAME_ID'] = df1['GAME_ID'].astype(str).str.strip()
    df1['PLAYER_ID'] = df1['PLAYER_ID'].astype(str).str.strip()
    df2['GAME_ID'] = df2['GAME_ID'].astype(str).str.strip()
    df2['PLAYER_ID'] = df2['player_id'].astype(str).str.strip()

    merged_df = pd.merge(df1, df2, on=['GAME_ID', 'PLAYER_ID'], suffixes=('_df1', '_df2'))

    merged_df['TIME_DIFF'] = (merged_df['MATCH_TIME_df1'] - merged_df['MATCH_TIME_df2']).abs()
    merged_df['DIST_DIFF'] = (merged_df['SHOT_DISTANCE'] - merged_df['SHOT_DIST']).abs()

    df2_made_mapped = merged_df['SHOT_RESULT'].astype(str).str.lower().str.strip() == 'made'
    merged_df['SHOT_MATCH_VALID'] = merged_df['SHOT_MADE_FLAG'].astype(int) == df2_made_mapped.astype(int)


    action_lower = merged_df['ACTION_TYPE'].astype(str).str.lower()
    merged_df['PHYSICAL_PENALTY'] = 0.0

    is_tip_action = action_lower.str.contains('tip|putback')
    bad_tip_match = is_tip_action & ((merged_df['TOUCH_TIME'] > 1.5) | (merged_df['DRIBBLES'] > 0))
    merged_df.loc[bad_tip_match, 'PHYSICAL_PENALTY'] += 20.0

    drive_keywords = ['driving', 'pullup', 'step', 'turnaround', 'fadeaway', 'running']
    is_drive_action = action_lower.str.contains('|'.join(drive_keywords))
    bad_drive_match = is_drive_action & (merged_df['TOUCH_TIME'] <= 1.2) & (merged_df['DRIBBLES'] == 0)
    merged_df.loc[bad_drive_match, 'PHYSICAL_PENALTY'] += 20.0

    merged_df['SMART_RANK_SCORE'] = (5 - merged_df['TIME_DIFF']) - merged_df['PHYSICAL_PENALTY']


    valid_mask = (
        (merged_df['TIME_DIFF'] <= 5) & 
        (merged_df['SHOT_MATCH_VALID']) & 
        (merged_df['DIST_DIFF'] <= 3)
    )
    final_merged_df = merged_df[valid_mask].copy()

    final_merged_df = final_merged_df.sort_values('SMART_RANK_SCORE', ascending=False)
    final_merged_df = final_merged_df.drop_duplicates(subset=['GAME_ID', 'PLAYER_ID', 'MATCH_TIME_df1'], keep='first')
    final_merged_df = final_merged_df.drop_duplicates(subset=['GAME_ID', 'PLAYER_ID', 'MATCH_TIME_df2'], keep='first')

    return final_merged_df

def get_player_detail(season):
    bio_stats = leaguedashplayerbiostats.LeagueDashPlayerBioStats(season=season)
    all_players_data = bio_stats.get_data_frames()[0]

    player_detail = all_players_data[['PLAYER_ID', 'PLAYER_HEIGHT_INCHES', 'PLAYER_WEIGHT', 'AGE', 'DRAFT_YEAR']].copy()
    player_detail.columns = ['PLAYER_ID', 'HEIGHT_INCHES', 'WEIGHT', 'SEASON_AGE', 'DRAFT_YEAR']
    player_detail = player_detail.dropna(subset=['HEIGHT_INCHES', 'WEIGHT']).copy()

    player_detail['DRAFT_YEAR'] = pd.to_numeric(player_detail['DRAFT_YEAR'], errors='coerce').fillna(2014)
    player_detail['EXP'] = 2014 - player_detail['DRAFT_YEAR']
    player_detail.loc[player_detail['EXP'] < 0, 'EXP'] = 0

    return player_detail

def add_features(df, season):
    df_feature = df.copy()
    player_detail = get_player_detail(season=season)

    player_detail['PLAYER_ID'] = player_detail['PLAYER_ID'].astype(str).str.strip()
    df_feature['PLAYER_ID'] = df_feature['PLAYER_ID'].astype(str).str.strip()
    df_feature['CLOSEST_DEFENDER_PLAYER_ID'] = df_feature['CLOSEST_DEFENDER_PLAYER_ID'].astype(str).str.strip()

    shooter_info = (
        player_detail[['PLAYER_ID', 'HEIGHT_INCHES', 'WEIGHT', 'SEASON_AGE', 'EXP']]
        .rename(columns={
            'HEIGHT_INCHES': 'SHOOTER_HEIGHT', 
            'WEIGHT': 'SHOOTER_WEIGHT'
        })
    )
    df_feature = pd.merge(df_feature, shooter_info, on='PLAYER_ID', how='left')

    defender_info = (
        player_detail[['PLAYER_ID', 'WEIGHT', 'HEIGHT_INCHES']]
        .rename(columns={
            'PLAYER_ID': 'CLOSEST_DEFENDER_PLAYER_ID',
            'WEIGHT': 'DEF_WEIGHT',
            'HEIGHT_INCHES': 'DEF_HEIGHT_INCHES'
        })
    )
    df_feature = pd.merge(df_feature, defender_info, on='CLOSEST_DEFENDER_PLAYER_ID', how='left')


    for col in ['SHOOTER_HEIGHT', 'DEF_HEIGHT_INCHES', 'SHOOTER_WEIGHT', 'DEF_WEIGHT', 'SEASON_AGE']:
        df_feature[col] = pd.to_numeric(df_feature[col], errors='coerce')
    df_feature['HEIGHT_DIFF'] = df_feature['SHOOTER_HEIGHT'] - df_feature['DEF_HEIGHT_INCHES']
    df_feature['WEIGHT_DIFF'] = df_feature['SHOOTER_WEIGHT'] - df_feature['DEF_WEIGHT']
    df_feature['PERIOD_TIME_REMAINING'] = (df_feature['MINUTES_REMAINING'].astype(int)*60) + df_feature['SECONDS_REMAINING'].astype(int)


    for col in ['SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'ACTION_TYPE', 'SHOT_TYPE']:
        df_feature[col] = df_feature[col].astype(str).str.strip()
    df_feature['IS_CORNER_3'] = df_feature['SHOT_ZONE_BASIC'].str.contains('Corner 3', case=False, na=False).astype(int)
    df_feature['IS_RESTRICTED_AREA'] = (df_feature['SHOT_ZONE_BASIC']=='Restricted Area').astype(int)
    df_feature['IS_JUMP_SHOT'] = df_feature['ACTION_TYPE'].str.contains('Jump Shot', case=False, na=False).astype(int)
    df_feature['IS_DUNK_OR_LAYUP'] = df_feature['ACTION_TYPE'].str.contains('Dunk|Layup', case=False, na=False).astype(int)
    df_feature['IS_CATCH_AND_SHOOT'] = ((df_feature['DRIBBLES']==0)&(df_feature['TOUCH_TIME']<=2)&(df_feature['IS_JUMP_SHOT']==1)).astype(int)


    df_feature['GAME_DATE'] = pd.to_datetime(df_feature['GAME_DATE'])
    df_feature = df_feature.sort_values(
        by=['GAME_DATE', 'GAME_ID', 'PERIOD_df1', 'PERIOD_TIME_REMAINING'], 
        ascending=[True, True, True, False]
    ).reset_index(drop=True)

    df_feature['PLAYER_PRIOR_ATT'] = df_feature.groupby(['GAME_ID', 'PLAYER_ID']).cumcount()

    df_feature['PLAYER_RFG_L5'] = (
        df_feature.groupby('PLAYER_ID')['SHOT_MADE_FLAG']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    ).fillna(0.45) 
    df_feature['TEAM_RFG_L10'] = (
        df_feature.groupby('TEAM_ID')['SHOT_MADE_FLAG']
        .transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    ).fillna(0.45)

    df_feature['FATIGUE_INDEX'] = df_feature['PLAYER_PRIOR_ATT'] * (df_feature['PERIOD_df1'] + (1 - df_feature['PERIOD_TIME_REMAINING']/720))
    df_feature['TEAM_MOMENTUM_X_MARGIN'] = df_feature['TEAM_RFG_L10'] * df_feature['FINAL_MARGIN']
    df_feature['SPATIAL_PRESSURE_IDX'] = df_feature['SHOT_DIST'] / (df_feature['CLOSE_DEF_DIST'] + 0.1)
    df_feature['PURE_CATCH_AND_SHOOT_GAP'] = (df_feature['DRIBBLES'] == 0).astype(int) * df_feature['CLOSE_DEF_DIST']
    df_feature['HOT_HAND_X_JUMP_SHOT'] = df_feature['PLAYER_RFG_L5'] * df_feature['IS_JUMP_SHOT']

    return df_feature

def cleaning(df):
    df_working = df.copy()
    
    columns_to_keep = [
    # IDs & Names
    'GAME_ID', 'GAME_EVENT_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME',
    
    # Game Context & Time
    'GAME_DATE', 'HTM', 'VTM', 'LOCATION', 'MATCH_TIME_df1', 'PERIOD_df1', 
    'PERIOD_TIME_REMAINING', 'FINAL_MARGIN',
    
    # Shot Details
    'GRID_TYPE', 'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 
    'SHOT_ZONE_RANGE', 'SHOT_DIST', 'LOC_X', 'LOC_Y', 
    'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME',
    
    # Outcomes
    'SHOT_MADE_FLAG',
    
    # Defender Data
    'CLOSEST_DEFENDER', 'CLOSEST_DEFENDER_PLAYER_ID', 'CLOSE_DEF_DIST',
    
    # Player Physicals & Mismatches
    'SHOOTER_HEIGHT', 'SHOOTER_WEIGHT', 'DEF_HEIGHT_INCHES', 'DEF_WEIGHT', 'HEIGHT_DIFF', 'WEIGHT_DIFF',
    
    # Dynamic Form Features
    'SEASON_AGE', 'EXP', 'PLAYER_PRIOR_ATT', 'PLAYER_RFG_L5', 'TEAM_RFG_L10',

    # Engineered Tactical Features
    'IS_CORNER_3', 'IS_RESTRICTED_AREA', 'IS_PAINT', 'IS_MID_RANGE', 'IS_BREAK_3',
    'IS_JUMP_SHOT', 'IS_DUNK_OR_LAYUP', 'IS_CATCH_AND_SHOOT',

    # Advanced Interaction Features
    'FATIGUE_INDEX', 'TEAM_MOMENTUM_X_MARGIN',

    'SPATIAL_PRESSURE_IDX', 'PURE_CATCH_AND_SHOOT_GAP', 'HOT_HAND_X_JUMP_SHOT'
]   

    df_cleaned = df_working[columns_to_keep].copy()
    df_cleaned = df_cleaned[df_cleaned['TOUCH_TIME'] > 0].copy()
    
    rename_dict = {
        'MATCH_TIME_df1': 'MATCH_TIME',
        'PERIOD_df1': 'PERIOD',
    }
    df_cleaned = df_cleaned.rename(columns=rename_dict)
    df_cleaned = df_cleaned.dropna()
    
    return df_cleaned

def run_process(season):
    df1_path = f'data/nba_api_{season}_shots.csv'
    #df2_path = f'data/shot_logs_{season}.csv'
    df2_path = 'data/shot_logs(in).csv'
    output_path = f'data/shotlog_{season}_preprocessed.csv'

    file_path = Path(output_path)
    if file_path.is_file():
        return pd.read_csv(output_path)

    print("🚀 Loading raw data...")
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    
    print("🔗 Merging shot datasets...")
    df_merged = merging(df1, df2)
    
    print("🧪 Engineering tactical & rolling features...")
    df_features = add_features(df_merged, season)

    print("📋 Cleaning dataset...")
    final_data = cleaning(df_features)

    print(f"💾 Exporting clean dataset to {output_path}...")
    final_data.to_csv(output_path, index=False)

    print(f"✅ Pipeline complete! Final shape: {final_data.shape}")
    return final_data

if __name__ == "__main__":
    season = '2014-15'
    data = run_process(season)
    print(data.head())
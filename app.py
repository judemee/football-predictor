from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
from io import StringIO
import requests
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

def normalize_columns(df):
    """Rename columns flexibly to standard names"""
    col_map = {}
    for col in df.columns:
        c = col.strip().lower().replace(" ", "").replace("_", "")
        if c in ["team", "club", "squad"]:
            col_map[col] = "team"
        elif c in ["wins", "w"]:
            col_map[col] = "wins"
        elif c in ["draws", "d"]:
            col_map[col] = "draws"
        elif c in ["losses", "l", "lost"]:
            col_map[col] = "losses"
        elif c in ["gf", "goalsfor", "for"]:
            col_map[col] = "goals_for"
        elif c in ["ga", "goalsagainst", "against"]:
            col_map[col] = "goals_against"
        elif c in ["gd", "goaldiff", "goaldifference"]:
            col_map[col] = "goal_diff"
        elif c in ["played", "matches", "games"]:
            col_map[col] = "played"
        elif c in ["points", "pts"]:
            col_map[col] = "points"
        elif c in ["season", "year"]:
            col_map[col] = "season"
        elif c in ["league", "competition"]:
            col_map[col] = "league"
    df = df.rename(columns=col_map)
    return df

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/load')
def load_csv():
    csv_url = request.args.get('csv_url')
    if not csv_url:
        return jsonify({'error': 'Missing CSV URL'}), 400

    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

        df = normalize_columns(df)
        if 'played' not in df.columns:
            # Auto-derive "played" from wins + draws + losses
            if all(c in df.columns for c in ['wins', 'draws', 'losses']):
                df['played'] = df['wins'] + df['draws'] + df['losses']
            else:
                df['played'] = np.nan

        if 'goal_diff' not in df.columns and all(c in df.columns for c in ['goals_for', 'goals_against']):
            df['goal_diff'] = df['goals_for'] - df['goals_against']

        expected = ['team', 'played', 'wins', 'draws', 'losses', 'goals_for', 'goals_against', 'goal_diff']
        for col in expected:
            if col not in df.columns:
                df[col] = np.nan

        teams = df.replace({np.nan: None}).to_dict(orient='records')
        return jsonify({'teams': teams})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    teamA = data.get('teamA')
    teamB = data.get('teamB')
    csv_url = data.get('csv_url')

    if not all([teamA, teamB, csv_url]):
        return jsonify({'error': 'Missing teamA, teamB, or csv_url'}), 400

    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df = normalize_columns(df)

        teamA_stats = df[df['team'].str.lower() == teamA.lower()].to_dict(orient='records')[0]
        teamB_stats = df[df['team'].str.lower() == teamB.lower()].to_dict(orient='records')[0]

        a_strength = (teamA_stats.get('wins', 0) + teamA_stats.get('goal_diff', 0) / 10)
        b_strength = (teamB_stats.get('wins', 0) + teamB_stats.get('goal_diff', 0) / 10)
        total = a_strength + b_strength + 1e-6

        probs = {
            'home': a_strength / total,
            'away': b_strength / total,
            'draw': 0.2
        }

        s = probs['home'] + probs['away'] + probs['draw']
        probs = {k: v / s for k, v in probs.items()}

        predicted = (
            teamA if probs['home'] > probs['away'] and probs['home'] > probs['draw']
            else teamB if probs['away'] > probs['home'] and probs['away'] > probs['draw']
            else 'Draw'
        )

        return jsonify({'predicted': predicted, 'probs': probs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8888))
    app.run(host='0.0.0.0', port=port, debug=True)

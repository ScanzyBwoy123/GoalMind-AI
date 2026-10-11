export default async ({ req, res, log, error }) => {
  const endpoint = (
    process.env.APPWRITE_FUNCTION_API_ENDPOINT ||
    "https://cloud.appwrite.io/v1"
  ).replace(/\/$/, "");

  const projectId = process.env.APPWRITE_FUNCTION_PROJECT_ID;
  const apiKey = process.env.APPWRITE_FUNCTION_API_KEY;

  const databaseId = "goalmind-football-db";
  const matchesTableId = "matches";
  const predictionsTableId = "predictions";

  if (!projectId || !apiKey) {
    return res.json({
      error: "The prediction service is not configured."
    }, 500);
  }

  let input;

  try {
    input = req.bodyJson || JSON.parse(req.body || "{}");
  } catch {
    return res.json({
      error: "Please provide valid JSON."
    }, 400);
  }

  const league = String(
    input.league || "English Premier League"
  ).trim();

  const homeTeam = String(input.home_team || "").trim();
  const awayTeam = String(input.away_team || "").trim();

  const matchDate = input.match_date
    ? new Date(input.match_date)
    : new Date();

  if (
    !homeTeam ||
    !awayTeam ||
    homeTeam.toLowerCase() === awayTeam.toLowerCase()
  ) {
    return res.json({
      error: "Enter two different teams."
    }, 400);
  }

  if (Number.isNaN(matchDate.getTime())) {
    return res.json({
      error: "The match date is invalid."
    }, 400);
  }

  const headers = {
    "X-Appwrite-Project": projectId,
    "X-Appwrite-Key": apiKey,
    "Content-Type": "application/json"
  };

  const normalize = name =>
    String(name || "")
      .toLowerCase()
      .replace(/\b(fc|afc|cf|sc)\b/g, "")
      .replace(/[^a-z0-9]+/g, " ")
      .trim();

  const sameTeam = (a, b) =>
    normalize(a) === normalize(b);

  const factorial = n => {
    let result = 1;

    for (let i = 2; i <= n; i++) {
      result *= i;
    }

    return result;
  };

  const poisson = (lambda, goals) =>
    Math.exp(-lambda) *
    Math.pow(lambda, goals) /
    factorial(goals);

  const clamp = (value, min, max) =>
    Math.max(min, Math.min(max, value));

  async function getMatches() {
    const matches = [];
    const pageSize = 100;

    for (let offset = 0; offset < 10000; offset += pageSize) {
      const url = new URL(
        `${endpoint}/tablesdb/${encodeURIComponent(databaseId)}` +
        `/tables/${encodeURIComponent(matchesTableId)}/rows`
      );

      url.searchParams.set("limit", String(pageSize));
      url.searchParams.set("offset", String(offset));

      const response = await fetch(url, { headers });

      if (!response.ok) {
        throw new Error(
          `Could not read historical matches: ${response.status}`
        );
      }

      const result = await response.json();
      const rows = result.rows || [];

            // Appwrite stores custom match fields inside each row's data object.
      // Flatten the rows so the prediction engine can read the fields.
      matches.push(...rows.map(row => ({ ...row, ...(row.data || {}) })));

      if (
        rows.length < pageSize ||
        matches.length >= (result.total || 0)
      ) {
        break;
      }
    }

    return matches;
  }

  function average(items, getter, fallback, weight = 5) {
    const total = items.reduce(
      (sum, item) => sum + getter(item),
      0
    );

    return (
      (total + fallback * weight) /
      (items.length + weight)
    );
  }

  function teamStatistics(team, history, avgHome, avgAway) {
    const homeMatches = history.filter(
      match => sameTeam(match.home_team, team)
    );

    const awayMatches = history.filter(
      match => sameTeam(match.away_team, team)
    );

    const homeAttack = average(
      homeMatches,
      match => match.home_goals,
      avgHome
    );

    const awayAttack = average(
      awayMatches,
      match => match.away_goals,
      avgAway
    );

    const homeConceded = average(
      homeMatches,
      match => match.away_goals,
      avgAway
    );

    const awayConceded = average(
      awayMatches,
      match => match.home_goals,
      avgHome
    );

    return {
      homeAttack: homeAttack / Math.max(avgHome, 0.1),
      awayAttack: awayAttack / Math.max(avgAway, 0.1),

      homeDefense:
        homeConceded / Math.max(avgAway, 0.1),

      awayDefense:
        awayConceded / Math.max(avgHome, 0.1)
    };
  }

  try {
    const allMatches = await getMatches();

    // Only use completed matches played before the requested date.
    const history = allMatches.filter(match =>
      match.league === league &&
      match.status === "completed" &&
      Number.isInteger(match.home_goals) &&
      Number.isInteger(match.away_goals) &&
      new Date(match.match_date) < matchDate
    );

    if (history.length < 40) {
      return res.json({
        error: "Not enough historical matches for this competition.",
        league,
        available_matches: history.length,
        minimum_required: 40
      }, 422);
    }

    const homeHistory = history.filter(match =>
      sameTeam(match.home_team, homeTeam) ||
      sameTeam(match.away_team, homeTeam)
    );

    const awayHistory = history.filter(match =>
      sameTeam(match.home_team, awayTeam) ||
      sameTeam(match.away_team, awayTeam)
    );

    if (!homeHistory.length || !awayHistory.length) {
      return res.json({
        error:
          "One or both teams were not found in the historical data.",
        home_team: homeTeam,
        away_team: awayTeam,
        league
      }, 422);
    }

    const avgHome =
      history.reduce(
        (sum, match) => sum + match.home_goals,
        0
      ) / history.length;

    const avgAway =
      history.reduce(
        (sum, match) => sum + match.away_goals,
        0
      ) / history.length;

    const homeStats = teamStatistics(
      homeTeam,
      homeHistory,
      avgHome,
      avgAway
    );

    const awayStats = teamStatistics(
      awayTeam,
      awayHistory,
      avgHome,
      avgAway
    );

    const expectedHome = clamp(
      avgHome *
      homeStats.homeAttack *
      awayStats.awayDefense,
      0.15,
      4.5
    );

    const expectedAway = clamp(
      avgAway *
      awayStats.awayAttack *
      homeStats.homeDefense,
      0.15,
      4.5
    );

    const maxGoals = 8;

    const homeDistribution = Array.from(
      { length: maxGoals + 1 },
      (_, goals) => poisson(expectedHome, goals)
    );

    const awayDistribution = Array.from(
      { length: maxGoals + 1 },
      (_, goals) => poisson(expectedAway, goals)
    );

    let homeWin = 0;
    let draw = 0;
    let awayWin = 0;
    let over25 = 0;
    let btts = 0;

    const scores = [];

    for (let homeGoals = 0; homeGoals <= maxGoals; homeGoals++) {
      for (let awayGoals = 0; awayGoals <= maxGoals; awayGoals++) {
        const probability =
          homeDistribution[homeGoals] *
          awayDistribution[awayGoals];

        if (homeGoals > awayGoals) {
          homeWin += probability;
        } else if (homeGoals === awayGoals) {
          draw += probability;
        } else {
          awayWin += probability;
        }

        if (homeGoals + awayGoals > 2) {
          over25 += probability;
        }

        if (homeGoals > 0 && awayGoals > 0) {
          btts += probability;
        }

        scores.push({
          score: `${homeGoals}-${awayGoals}`,
          probability
        });
      }
    }

    const outcomeTotal = homeWin + draw + awayWin;

    homeWin /= outcomeTotal;
    draw /= outcomeTotal;
    awayWin /= outcomeTotal;

    const topScores = scores
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5)
      .map(item => ({
        score: item.score,
        probability: Number(
          (item.probability * 100).toFixed(1)
        )
      }));

    const prediction = {
      match_date: matchDate.toISOString(),
      league,
      home_team: homeTeam,
      away_team: awayTeam,

      home_win_prob: Number(homeWin.toFixed(4)),
      draw_prob: Number(draw.toFixed(4)),
      away_win_prob: Number(awayWin.toFixed(4)),

      over25_prob: Number(over25.toFixed(4)),
      btts_prob: Number(btts.toFixed(4)),

      predicted_home_goals: Number(expectedHome.toFixed(2)),
      predicted_away_goals: Number(expectedAway.toFixed(2)),

      predicted_score: topScores[0].score,
      top_scores: topScores,

      history_matches_used: history.length,
      model_version: "goalmind-poisson-0.1",

      note:
        "Experimental statistical estimates. " +
        "Predictions are not guarantees."
    };

    // Save the result to the Appwrite predictions table.
    const saveUrl =
      `${endpoint}/tablesdb/${encodeURIComponent(databaseId)}` +
      `/tables/${encodeURIComponent(predictionsTableId)}/rows`;

    const saveResponse = await fetch(saveUrl, {
      method: "POST",
      headers,
      body: JSON.stringify({
        rowId: crypto.randomUUID(),

        data: {
          match_date: prediction.match_date,
          league,
          home_team: homeTeam,
          away_team: awayTeam,

          home_win_prob: prediction.home_win_prob,
          draw_prob: prediction.draw_prob,
          away_win_prob: prediction.away_win_prob,

          over25_prob: prediction.over25_prob,
          btts_prob: prediction.btts_prob,

          predicted_home_goals:
            prediction.predicted_home_goals,

          predicted_away_goals:
            prediction.predicted_away_goals,

          predicted_score: prediction.predicted_score,
          model_version: prediction.model_version,
          created_at: new Date().toISOString()
        }
      })
    });

    if (!saveResponse.ok) {
      const saveError = await saveResponse.text();

      log(`Prediction calculated but not saved: ${saveError}`);

      prediction.saved = false;
    } else {
      prediction.saved = true;
    }

    return res.json(prediction);

  } catch (exception) {
    error(String(exception.stack || exception));

    return res.json({
      error: "The prediction could not be completed.",
      detail: String(exception.message || exception)
    }, 500);
  }
};

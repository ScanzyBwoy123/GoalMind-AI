"use strict";
/**
 * GoalMind AI — Offline Model Backtesting Tool
 *
 * Models:
 *   1. Poisson
 *   2. Elo
 *   3. Hybrid (Poisson + Elo)
 *
 * Input:
 *   A local JSON or CSV file containing historical match results.
 *
 * IMPORTANT:
 *   - No Appwrite connection.
 *   - No API keys required.
 *   - No changes to the live prediction function.
 *   - Uses chronological evaluation to avoid training on future results.
 *
 * Run:
 *   node tools/model-backtest.js matches.json
 *
 * Supported JSON format:
 * [
 *   {
 *     "date": "2025-01-10",
 *     "homeTeam": "Arsenal",
 *     "awayTeam": "Chelsea",
 *     "homeGoals": 2,
 *     "awayGoals": 1
 *   }
 * ]
 *
 * Supported CSV headers:
 * date,homeTeam,awayTeam,homeGoals,awayGoals
 */
const fs = require("node:fs");
const path = require("node:path");
const CONFIG = {
  minimumTrainingMatches: 30,
  initialElo: 1500,
  eloK: 24,
  homeAdvantageElo: 60,
  homeAdvantageGoals: 0.25,
  poissonMaximumGoals: 10,
  hybridEloWeight: 0.35,
  hybridPoissonWeight: 0.65
};
function fail(message) {
  console.error(`\nERROR: ${message}\n`);
  process.exitCode = 1;
}
function number(value) {
  if (value === "" || value === null || value === undefined) {
    return NaN;
  }
  return Number(value);
}
function normalizeDate(value) {
  const raw = String(value || "").trim();

  // Supports Football-Data dates such as 19/10/26.
  const ukDate = raw.match(
    /^(\d{1,2})\/(\d{1,2})\/(\d{2}|\d{4})$/
  );

  if (ukDate) {
    const day = Number(ukDate[1]);
    const month = Number(ukDate[2]);
    let year = Number(ukDate[3]);

    if (year < 100) {
      year += year >= 50 ? 1900 : 2000;
    }

    const parsed = new Date(Date.UTC(year, month - 1, day));

    if (
      parsed.getUTCFullYear() !== year ||
      parsed.getUTCMonth() !== month - 1 ||
      parsed.getUTCDate() !== day
    ) {
      return null;
    }

    return parsed.toISOString();
  }

  const timestamp = Date.parse(raw);

  return Number.isFinite(timestamp)
    ? new Date(timestamp).toISOString()
    : null;
}

function normalizeMatch(row, index) {
  const date =
    row.date ||
    row.match_date ||
    row.matchDate ||
    row.Date;

  const homeTeam =
    row.homeTeam ||
    row.home_team ||
    row.home ||
    row.HomeTeam;

  const awayTeam =
    row.awayTeam ||
    row.away_team ||
    row.away ||
    row.AwayTeam;

  const homeGoals = number(
    row.homeGoals ??
    row.home_goals ??
    row.home_score ??
    row.FTHG
  );

  const awayGoals = number(
    row.awayGoals ??
    row.away_goals ??
    row.away_score ??
    row.FTAG
  );

  const normalizedDate = normalizeDate(date);

  if (!normalizedDate || !homeTeam || !awayTeam) {
    throw new Error(
      `Match ${index + 1} is missing a valid date or team name.`
    );
  }

  if (
    !Number.isInteger(homeGoals) ||
    !Number.isInteger(awayGoals) ||
    homeGoals < 0 ||
    awayGoals < 0
  ) {
    throw new Error(
      `Match ${index + 1} has an invalid or missing score.`
    );
  }

  if (
    String(homeTeam).trim().toLowerCase() ===
    String(awayTeam).trim().toLowerCase()
  ) {
    throw new Error(
      `Match ${index + 1} has the same home and away team.`
    );
  }

  return {
    date: normalizedDate,
    homeTeam: String(homeTeam).trim(),
    awayTeam: String(awayTeam).trim(),
    homeGoals,
    awayGoals
  };
}
function parseCSV(text) {
  const lines = text
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .filter(line => line.trim() !== "");
  if (lines.length < 2) {
    throw new Error("The CSV file contains no match rows.");
  }
  function parseLine(line) {
    const fields = [];
    let field = "";
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        if (quoted && line[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          quoted = !quoted;
        }
      } else if (char === "," && !quoted) {
        fields.push(field.trim());
        field = "";
      } else {
        field += char;
      }
    }
    if (quoted) {
      throw new Error("CSV contains an unclosed quotation mark.");
    }
    fields.push(field.trim());
    return fields;
  }
  const headers = parseLine(lines[0]).map(header =>
    header.replace(/^\uFEFF/, "").trim()
  );
  return lines.slice(1).map(line => {
    const values = parseLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    return row;
  });
}
function loadMatches(filePath) {
  const resolved = path.resolve(filePath);
  if (!fs.existsSync(resolved)) {
    throw new Error(`Dataset not found: ${resolved}`);
  }
  const content = fs.readFileSync(resolved, "utf8");
  let rows;
  if (resolved.toLowerCase().endsWith(".csv")) {
    rows = parseCSV(content);
  } else if (resolved.toLowerCase().endsWith(".json")) {
    rows = JSON.parse(content);
    if (!Array.isArray(rows)) {
      throw new Error("The JSON dataset must contain an array of matches.");
    }
  } else {
    throw new Error("Dataset must be a .json or .csv file.");
  }
  const matches = rows.map(normalizeMatch);
  matches.sort((a, b) =>
    a.date.localeCompare(b.date)
  );
  for (let i = 1; i < matches.length; i++) {
    if (matches[i].date < matches[i - 1].date) {
      throw new Error("Dataset sorting failed.");
    }
  }
  return matches;
}
function poissonProbability(goals, lambda) {
  if (lambda <= 0) {
    return goals === 0 ? 1 : 0;
  }
  let probability = Math.exp(-lambda);
  for (let i = 1; i <= goals; i++) {
    probability *= lambda / i;
  }
  return probability;
}
function poissonDistribution(lambda, maxGoals) {
  const values = [];
  for (let goals = 0; goals <= maxGoals; goals++) {
    values.push(poissonProbability(goals, lambda));
  }
  // Include the small probability mass beyond maxGoals.
  // Renormalization keeps the distribution usable for comparisons.
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map(value => value / total);
}
function expectedGoals(trainingMatches) {
  let homeGoals = 0;
  let awayGoals = 0;
  for (const match of trainingMatches) {
    homeGoals += match.homeGoals;
    awayGoals += match.awayGoals;
  }
  const count = trainingMatches.length || 1;
  return {
    home: homeGoals / count,
    away: awayGoals / count
  };
}
function poissonPrediction(trainingMatches, target) {
  const league = expectedGoals(trainingMatches);
  const homeStats = {};
  const awayStats = {};
  function getStats(store, team) {
    if (!store[team]) {
      store[team] = {
        goalsFor: 0,
        goalsAgainst: 0,
        matches: 0
      };
    }
    return store[team];
  }
  for (const match of trainingMatches) {
    const home = getStats(homeStats, match.homeTeam);
    home.goalsFor += match.homeGoals;
    home.goalsAgainst += match.awayGoals;
    home.matches++;
    const away = getStats(awayStats, match.awayTeam);
    away.goalsFor += match.awayGoals;
    away.goalsAgainst += match.homeGoals;
    away.matches++;
  }
  function rates(store, team, baseline) {
    const stats = store[team];
    if (!stats || stats.matches === 0) {
      return {
        attack: 1,
        defense: 1
      };
    }
    const attack = stats.goalsFor / stats.matches;
    const defense = stats.goalsAgainst / stats.matches;
    return {
      attack: attack / Math.max(baseline, 0.1),
      defense: defense / Math.max(baseline, 0.1)
    };
  }
  const homeAttack = rates(
    homeStats,
    target.homeTeam,
    league.home
  );
  const homeDefense = rates(
    homeStats,
    target.homeTeam,
    league.away
  );
  const awayAttack = rates(
    awayStats,
    target.awayTeam,
    league.away
  );
  const awayDefense = rates(
    awayStats,
    target.awayTeam,
    league.home
  );
  const homeLambda = Math.max(
    0.1,
    league.home *
      homeAttack.attack *
      awayDefense.defense +
      CONFIG.homeAdvantageGoals
  );
  const awayLambda = Math.max(
    0.1,
    league.away *
      awayAttack.attack *
      homeDefense.defense
  );
  return {
    homeLambda,
    awayLambda,
    homeProbabilities: poissonDistribution(
      homeLambda,
      CONFIG.poissonMaximumGoals
    ),
    awayProbabilities: poissonDistribution(
      awayLambda,
      CONFIG.poissonMaximumGoals
    )
  };
}
function getElo(ratings, team) {
  if (!(team in ratings)) {
    ratings[team] = CONFIG.initialElo;
  }
  return ratings[team];
}
function updateElo(ratings, match) {
  const home = getElo(ratings, match.homeTeam);
  const away = getElo(ratings, match.awayTeam);
  const expectedHome =
    1 /
    (1 +
      Math.pow(
        10,
        (away - (home + CONFIG.homeAdvantageElo)) / 400
      ));
  const expectedAway = 1 - expectedHome;
  let actualHome;
  if (match.homeGoals > match.awayGoals) {
    actualHome = 1;
  } else if (match.homeGoals < match.awayGoals) {
    actualHome = 0;
  } else {
    actualHome = 0.5;
  }
  const actualAway = 1 - actualHome;
  // A modest margin-of-victory adjustment.
  const goalDifference = Math.abs(
    match.homeGoals - match.awayGoals
  );
  const marginMultiplier =
    goalDifference > 1
      ? Math.log(goalDifference + 1) * 2.2 /
        (0.001 * Math.abs(home - away) + 2.2)
      : 1;
  ratings[match.homeTeam] =
    home +
    CONFIG.eloK *
      marginMultiplier *
      (actualHome - expectedHome);
  ratings[match.awayTeam] =
    away +
    CONFIG.eloK *
      marginMultiplier *
      (actualAway - expectedAway);
}
function eloPrediction(ratings, target, trainingMatches) {
  const homeRating = getElo(ratings, target.homeTeam);
  const awayRating = getElo(ratings, target.awayTeam);
  // Estimate the relative strength of the home team.
  const homeStrength =
    1 /
    (1 +
      Math.pow(
        10,
        (awayRating -
          (homeRating + CONFIG.homeAdvantageElo)) / 400
      ));
  // Estimate the draw rate using historical matches only.
  let draws = 0;
  for (const match of trainingMatches) {
    if (match.homeGoals === match.awayGoals) {
      draws++;
    }
  }
  // Smooth the draw estimate to avoid extreme probabilities.
  const drawProbability =
    (draws + 1) / (trainingMatches.length + 4);
  // Divide the remaining probability between home and away wins.
  const decisiveProbability = 1 - drawProbability;
  return {
    home: decisiveProbability * homeStrength,
    draw: drawProbability,
    away: decisiveProbability * (1 - homeStrength)
  };
}
function outcomeFromScores(homeGoals, awayGoals) {
  if (homeGoals > awayGoals) return "H";
  if (homeGoals < awayGoals) return "A";
  return "D";
}
function outcomeProbabilities(prediction) {
  let home = 0;
  let draw = 0;
  let away = 0;
  for (
    let h = 0;
    h < prediction.homeProbabilities.length;
    h++
  ) {
    for (
      let a = 0;
      a < prediction.awayProbabilities.length;
      a++
    ) {
      const probability =
        prediction.homeProbabilities[h] *
        prediction.awayProbabilities[a];
      if (h > a) home += probability;
      else if (h < a) away += probability;
      else draw += probability;
    }
  }
  const total = home + draw + away;
  return {
    home: home / total,
    draw: draw / total,
    away: away / total
  };
}
function hybridPrediction(poisson, elo) {
  const poissonOutcomes = outcomeProbabilities(poisson);
  const pWeight = CONFIG.hybridPoissonWeight;
  const eWeight = CONFIG.hybridEloWeight;
  const combined = {
    home:
      pWeight * poissonOutcomes.home +
      eWeight * elo.home,
    draw:
      pWeight * poissonOutcomes.draw +
      eWeight * elo.draw,
    away:
      pWeight * poissonOutcomes.away +
      eWeight * elo.away
  };
  const total = combined.home + combined.draw + combined.away;
  return {
    home: combined.home / total,
    draw: combined.draw / total,
    away: combined.away / total
  };
}
function brierScore(probabilities, actual) {
  const actualValues = {
    home: actual === "H" ? 1 : 0,
    draw: actual === "D" ? 1 : 0,
    away: actual === "A" ? 1 : 0
  };
  return (
    Math.pow(probabilities.home - actualValues.home, 2) +
    Math.pow(probabilities.draw - actualValues.draw, 2) +
    Math.pow(probabilities.away - actualValues.away, 2)
  ) / 3;
}
function logLoss(probabilities, actual) {
  const chosen =
    actual === "H"
      ? probabilities.home
      : actual === "D"
      ? probabilities.draw
      : probabilities.away;
  return -Math.log(Math.max(chosen, 1e-15));
}
function createMetrics() {
  return {
    correct: 0,
    brierTotal: 0,
    logLossTotal: 0,
    matches: 0
  };
}
function addResult(metrics, prediction, actual, label = "Model") {
  const probabilities = [
    prediction.home,
    prediction.draw,
    prediction.away
  ];

  if (
    probabilities.some(
      p => !Number.isFinite(p) || p < 0 || p > 1
    )
  ) {
    throw new Error(
      `${label} produced an invalid probability.`
    );
  }

  const total = probabilities.reduce(
    (sum, p) => sum + p,
    0
  );

  if (Math.abs(total - 1) > 1e-8) {
    throw new Error(
      `${label} probabilities do not add up to 1.`
    );
  }

  const choices = [
    ["H", prediction.home],
    ["D", prediction.draw],
    ["A", prediction.away]
  ];

  choices.sort((a, b) => b[1] - a[1]);

  if (choices[0][0] === actual) {
    metrics.correct++;
  }

  metrics.brierTotal += brierScore(prediction, actual);
  metrics.logLossTotal += logLoss(prediction, actual);
  metrics.matches++;
}
function formatMetrics(name, metrics) {
  if (!metrics.matches) {
    return {
      model: name,
      matches: 0,
      accuracy: "N/A",
      brierScore: "N/A",
      logLoss: "N/A"
    };
  }
  return {
    model: name,
    matches: metrics.matches,
    accuracy:
      ((metrics.correct / metrics.matches) * 100).toFixed(2) + "%",
    brierScore:
      (metrics.brierTotal / metrics.matches).toFixed(4),
    logLoss:
      (metrics.logLossTotal / metrics.matches).toFixed(4)
  };
}
function runBacktest(matches) {
  if (
    matches.length <
    CONFIG.minimumTrainingMatches + 1
  ) {
    throw new Error(
      `At least ${CONFIG.minimumTrainingMatches + 1} matches are required. ` +
      `Found ${matches.length}.`
    );
  }

  // Keep matches played on the same date in the same group.
  const cutoffDate =
    matches[CONFIG.minimumTrainingMatches - 1].date;

  const trainingMatches = matches.filter(
    match => match.date <= cutoffDate
  );

  const testMatches = matches.filter(
    match => match.date > cutoffDate
  );

  if (
    trainingMatches.length < CONFIG.minimumTrainingMatches ||
    testMatches.length === 0
  ) {
    throw new Error(
      "Could not create a chronological train/test split."
    );
  }

  const ratings = {};

  const metrics = {
    Poisson: createMetrics(),
    Elo: createMetrics(),
    Hybrid: createMetrics()
  };

  // Initialize Elo using training matches only.
  for (const match of trainingMatches) {
    updateElo(ratings, match);
  }

  let dateGroup = [];

  function evaluateDateGroup(group) {
    if (group.length === 0) return;

    const matchDate = group[0].date;

    // Only use matches from earlier dates.
    const priorMatches = matches.filter(
      match => match.date < matchDate
    );

    // Predict every match on this date before updating Elo.
    for (const match of group) {
      const poisson = poissonPrediction(
        priorMatches,
        match
      );

      const poissonOutcomes =
        outcomeProbabilities(poisson);

      const elo = eloPrediction(
        ratings,
        match,
        priorMatches
      );

      const hybrid = hybridPrediction(
        poisson,
        elo
      );

      const actual = outcomeFromScores(
        match.homeGoals,
        match.awayGoals
      );

      addResult(
        metrics.Poisson,
        poissonOutcomes,
        actual,
        "Poisson"
      );

      addResult(
        metrics.Elo,
        elo,
        actual,
        "Elo"
      );

      addResult(
        metrics.Hybrid,
        hybrid,
        actual,
        "Hybrid"
      );
    }

    // Update Elo only after all predictions for this date.
    for (const match of group) {
      updateElo(ratings, match);
    }
  }

  let currentDate = null;

  for (const match of testMatches) {
    if (
      currentDate !== null &&
      match.date !== currentDate
    ) {
      evaluateDateGroup(dateGroup);
      dateGroup = [];
    }

    dateGroup.push(match);
    currentDate = match.date;
  }

  evaluateDateGroup(dateGroup);

  const results = [
    formatMetrics("Poisson", metrics.Poisson),
    formatMetrics("Elo", metrics.Elo),
    formatMetrics("Hybrid", metrics.Hybrid)
  ];

  return {
    totalMatches: matches.length,
    trainingMatches: trainingMatches.length,
    testedMatches: testMatches.length,
    results
  };
}
function main() {
  const input = process.argv[2];
  if (!input) {
    console.log(`
GoalMind AI — Offline Model Backtesting
Usage:
  node tools/model-backtest.js path/to/matches.json
  node tools/model-backtest.js path/to/matches.csv
Required data:
  date, homeTeam, awayTeam, homeGoals, awayGoals
This script does not connect to Appwrite or modify your live site.
`);
    return;
  }
  const matches = loadMatches(input);
  const report = runBacktest(matches);
  console.log("\n======================================");
  console.log("       GOALMIND AI BACKTEST REPORT");
  console.log("======================================");
  console.log(`Historical matches: ${report.totalMatches}`);
  console.log(`Initial training:   ${report.trainingMatches}`);
  console.log(`Matches evaluated:  ${report.testedMatches}`);
  console.log("\nRESULTS");
  console.table(report.results);
  const best = [...report.results]
    .filter(row => row.matches > 0)
    .sort((a, b) => Number(a.brierScore) - Number(b.brierScore))[0];
  if (best) {
    console.log(
      `\nLowest Brier score: ${best.model}. ` +
      "Lower is better for probabilistic forecasts."
    );
  }
  console.log(
    "\nImportant: this is an initial baseline, not proof of predictive superiority."
  );
}
try {
  main();
} catch (error) {
  fail(error.message);
}

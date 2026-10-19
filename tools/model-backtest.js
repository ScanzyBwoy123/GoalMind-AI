"use strict";

/**
 * GoalMind AI — Multi-Season Model Backtesting
 *
 * Models:
 *   1. Poisson
 *   2. Elo
 *   3. Hybrid (Poisson + Elo)
 *
 * Usage:
 *   node tools/model-backtest.js season.csv
 *
 *   node tools/model-backtest.js season1.csv season2.csv season3.csv season4.csv
 *
 * Supported CSV columns:
 *   Date,HomeTeam,AwayTeam,FTHG,FTAG
 *
 * Also supports JSON arrays containing:
 *   date, homeTeam, awayTeam, homeGoals, awayGoals
 *
 * This is an offline testing tool. It does not connect to
 * Appwrite or modify live predictions.
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
  hybridPoissonWeight: 0.65,
  calibrationBins: 10
};

const OUTCOMES = ["H", "D", "A"];

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

  const match = raw.match(
    /^(\d{1,2})\/(\d{1,2})\/(\d{2}|\d{4})$/
  );

  if (match) {
    const day = Number(match[1]);
    const month = Number(match[2]);
    let year = Number(match[3]);

    if (year < 100) {
      year += year >= 50 ? 1900 : 2000;
    }

    const date = new Date(Date.UTC(year, month - 1, day));

    if (
      date.getUTCFullYear() !== year ||
      date.getUTCMonth() !== month - 1 ||
      date.getUTCDate() !== day
    ) {
      return null;
    }

    return date.toISOString().slice(0, 10);
  }

  const timestamp = Date.parse(raw);

  return Number.isFinite(timestamp)
    ? new Date(timestamp).toISOString().slice(0, 10)
    : null;
}

function normalizeMatch(row, index) {
  const date = normalizeDate(
    row.date || row.match_date || row.matchDate || row.Date
  );

  const homeTeam = String(
    row.homeTeam || row.home_team || row.home || row.HomeTeam || ""
  ).trim();

  const awayTeam = String(
    row.awayTeam || row.away_team || row.away || row.AwayTeam || ""
  ).trim();

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

  if (!date || !homeTeam || !awayTeam) {
    throw new Error(
      `Match ${index + 1} has a missing date or team name.`
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

  if (homeTeam.toLowerCase() === awayTeam.toLowerCase()) {
    throw new Error(
      `Match ${index + 1} has the same home and away team.`
    );
  }

  return {
    date,
    homeTeam,
    awayTeam,
    homeGoals,
    awayGoals
  };
}

function parseCSVLine(line) {
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

function parseCSV(content) {
  const lines = content
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .filter(line => line.trim() !== "");

  if (lines.length < 2) {
    throw new Error("CSV contains no match rows.");
  }

  const headers = parseCSVLine(lines[0]).map(header =>
    header.replace(/^\uFEFF/, "").trim()
  );

  return lines.slice(1).map(line => {
    const values = parseCSVLine(line);
    const row = {};

    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });

    return row;
  });
}

function loadMatches(filePaths) {
  const matches = [];

  for (const filePath of filePaths) {
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
        throw new Error(
          `JSON dataset must contain an array: ${resolved}`
        );
      }
    } else {
      throw new Error(
        `Unsupported file: ${resolved}. Use CSV or JSON.`
      );
    }

    rows.forEach((row, index) => {
      matches.push(normalizeMatch(row, index));
    });
  }

  matches.sort((a, b) => a.date.localeCompare(b.date));

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

function poissonDistribution(lambda, maximumGoals) {
  const values = [];

  for (let goals = 0; goals <= maximumGoals; goals++) {
    values.push(poissonProbability(goals, lambda));
  }

  const total = values.reduce((sum, value) => sum + value, 0);

  return values.map(value => value / total);
}

function expectedGoals(matches) {
  if (matches.length === 0) {
    return { home: 1.4, away: 1.1 };
  }

  let homeGoals = 0;
  let awayGoals = 0;

  for (const match of matches) {
    homeGoals += match.homeGoals;
    awayGoals += match.awayGoals;
  }

  return {
    home: homeGoals / matches.length,
    away: awayGoals / matches.length
  };
}

function poissonPrediction(matches, target) {
  const league = expectedGoals(matches);
  const homeStats = {};
  const awayStats = {};

  function statsFor(store, team) {
    if (!store[team]) {
      store[team] = {
        goalsFor: 0,
        goalsAgainst: 0,
        matches: 0
      };
    }

    return store[team];
  }

  for (const match of matches) {
    const home = statsFor(homeStats, match.homeTeam);

    home.goalsFor += match.homeGoals;
    home.goalsAgainst += match.awayGoals;
    home.matches++;

    const away = statsFor(awayStats, match.awayTeam);

    away.goalsFor += match.awayGoals;
    away.goalsAgainst += match.homeGoals;
    away.matches++;
  }

  function rates(store, team, baseline) {
    const stats = store[team];

    if (!stats || stats.matches === 0) {
      return { attack: 1, defense: 1 };
    }

    return {
      attack:
        (stats.goalsFor / stats.matches) /
        Math.max(baseline, 0.1),

      defense:
        (stats.goalsAgainst / stats.matches) /
        Math.max(baseline, 0.1)
    };
  }

  const homeAttack = rates(homeStats, target.homeTeam, league.home);
  const homeDefense = rates(homeStats, target.homeTeam, league.away);
  const awayAttack = rates(awayStats, target.awayTeam, league.away);
  const awayDefense = rates(awayStats, target.awayTeam, league.home);

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

function outcomeProbabilities(prediction) {
  let home = 0;
  let draw = 0;
  let away = 0;

  for (let h = 0; h < prediction.homeProbabilities.length; h++) {
    for (let a = 0; a < prediction.awayProbabilities.length; a++) {
      const probability =
        prediction.homeProbabilities[h] *
        prediction.awayProbabilities[a];

      if (h > a) {
        home += probability;
      } else if (h < a) {
        away += probability;
      } else {
        draw += probability;
      }
    }
  }

  const total = home + draw + away;

  return {
    H: home / total,
    D: draw / total,
    A: away / total
  };
}

function getElo(ratings, team) {
  return team in ratings
    ? ratings[team]
    : CONFIG.initialElo;
}

function eloPrediction(ratings, target, matches) {
  const homeRating = getElo(ratings, target.homeTeam);
  const awayRating = getElo(ratings, target.awayTeam);

  const homeStrength =
    1 /
    (
      1 +
      Math.pow(
        10,
        (
          awayRating -
          (homeRating + CONFIG.homeAdvantageElo)
        ) / 400
      )
    );

  let draws = 0;

  for (const match of matches) {
    if (match.homeGoals === match.awayGoals) {
      draws++;
    }
  }

  const drawProbability =
    (draws + 1) / (matches.length + 4);

  const decisiveProbability = 1 - drawProbability;

  return {
    H: decisiveProbability * homeStrength,
    D: drawProbability,
    A: decisiveProbability * (1 - homeStrength)
  };
}

function updateEloGroup(ratings, group) {
  /*
   * All predictions for the date must happen before this function.
   * Every match update uses the ratings snapshot from the start
   * of the date, avoiding match-order influence within that date.
   */
  const changes = {};

  function rating(team) {
    return team in ratings
      ? ratings[team]
      : CONFIG.initialElo;
  }

  function addChange(team, change) {
    changes[team] = (changes[team] || 0) + change;
  }

  for (const match of group) {
    const home = rating(match.homeTeam);
    const away = rating(match.awayTeam);

    const expectedHome =
      1 /
      (
        1 +
        Math.pow(
          10,
          (away - (home + CONFIG.homeAdvantageElo)) / 400
        )
      );

    const actualHome =
      match.homeGoals > match.awayGoals ? 1 :
      match.homeGoals < match.awayGoals ? 0 : 0.5;

    const difference = Math.abs(home - away);
    const goalDifference = Math.abs(
      match.homeGoals - match.awayGoals
    );

    const marginMultiplier =
      goalDifference > 1
        ? Math.log(goalDifference + 1) * 2.2 /
          (0.001 * difference + 2.2)
        : 1;

    const change =
      CONFIG.eloK *
      marginMultiplier *
      (actualHome - expectedHome);

    addChange(match.homeTeam, change);
    addChange(match.awayTeam, -change);
  }

  for (const [team, change] of Object.entries(changes)) {
    ratings[team] = rating(team) + change;
  }
}

function hybridPrediction(poisson, elo) {
  const pWeight = CONFIG.hybridPoissonWeight;
  const eWeight = CONFIG.hybridEloWeight;

  const combined = {};

  for (const outcome of OUTCOMES) {
    combined[outcome] =
      pWeight * poisson[outcome] +
      eWeight * elo[outcome];
  }

  const total = OUTCOMES.reduce(
    (sum, outcome) => sum + combined[outcome],
    0
  );

  for (const outcome of OUTCOMES) {
    combined[outcome] /= total;
  }

  return combined;
}

function outcomeFromScores(homeGoals, awayGoals) {
  if (homeGoals > awayGoals) return "H";
  if (homeGoals < awayGoals) return "A";
  return "D";
}

function createMetrics() {
  return {
    correct: 0,
    brierTotal: 0,
    logLossTotal: 0,
    matches: 0,
    calibration: {
      H: [],
      D: [],
      A: []
    }
  };
}

function addResult(metrics, prediction, actual) {
  const total = OUTCOMES.reduce(
    (sum, outcome) => sum + prediction[outcome],
    0
  );

  if (
    OUTCOMES.some(outcome => {
      const probability = prediction[outcome];

      return (
        !Number.isFinite(probability) ||
        probability < 0 ||
        probability > 1
      );
    }) ||
    Math.abs(total - 1) > 1e-8
  ) {
    throw new Error("A model produced invalid probabilities.");
  }

  const predictedOutcome = OUTCOMES.reduce(
    (best, outcome) =>
      prediction[outcome] > prediction[best] ? outcome : best,
    "H"
  );

  if (predictedOutcome === actual) {
    metrics.correct++;
  }

  let brier = 0;

  for (const outcome of OUTCOMES) {
    const observed = outcome === actual ? 1 : 0;
    const probability = prediction[outcome];

    brier += Math.pow(probability - observed, 2);

    metrics.calibration[outcome].push({
      predicted: probability,
      observed
    });
  }

  // Multiclass Brier score is divided by the three outcomes.
  metrics.brierTotal += brier / 3;

  metrics.logLossTotal += -Math.log(
    Math.max(prediction[actual], 1e-15)
  );

  metrics.matches++;
}

function formatMetrics(name, metrics) {
  return {
    Model: name,
    Matches: metrics.matches,
    Accuracy: metrics.matches
      ? `${(100 * metrics.correct / metrics.matches).toFixed(2)}%`
      : "N/A",
    "Brier Score": metrics.matches
      ? (metrics.brierTotal / metrics.matches).toFixed(4)
      : "N/A",
    "Log Loss": metrics.matches
      ? (metrics.logLossTotal / metrics.matches).toFixed(4)
      : "N/A"
  };
}

function evaluateDateGroup(group, priorMatches, ratings, metrics) {
  /*
   * No match on this date is added to the history until all
   * predictions for the date have been generated.
   */
  for (const match of group) {
    const poisson = outcomeProbabilities(
      poissonPrediction(priorMatches, match)
    );

    const elo = eloPrediction(ratings, match, priorMatches);

    const hybrid = hybridPrediction(poisson, elo);

    const actual = outcomeFromScores(
      match.homeGoals,
      match.awayGoals
    );

    addResult(metrics.Poisson, poisson, actual);
    addResult(metrics.Elo, elo, actual);
    addResult(metrics.Hybrid, hybrid, actual);
  }

  updateEloGroup(ratings, group);
}

function runBacktest(matches) {
  if (matches.length <= CONFIG.minimumTrainingMatches) {
    throw new Error(
      `At least ${CONFIG.minimumTrainingMatches + 1} matches are required.`
    );
  }

  /*
   * Select a cutoff after the minimum number of matches.
   * All matches on the cutoff date remain in training so that
   * no date is split between training and testing.
   */
  const cutoffDate =
    matches[CONFIG.minimumTrainingMatches - 1].date;

  const trainingMatches = matches.filter(
    match => match.date <= cutoffDate
  );

  const testMatches = matches.filter(
    match => match.date > cutoffDate
  );

  if (testMatches.length === 0) {
    throw new Error("No test matches remain after the training cutoff.");
  }

  const metrics = {
    Poisson: createMetrics(),
    Elo: createMetrics(),
    Hybrid: createMetrics()
  };

  const ratings = {};

  // Train Elo in chronological date groups.
  const trainingByDate = new Map();

  for (const match of trainingMatches) {
    if (!trainingByDate.has(match.date)) {
      trainingByDate.set(match.date, []);
    }

    trainingByDate.get(match.date).push(match);
  }

  for (const group of trainingByDate.values()) {
    updateEloGroup(ratings, group);
  }

  const testByDate = new Map();

  for (const match of testMatches) {
    if (!testByDate.has(match.date)) {
      testByDate.set(match.date, []);
    }

    testByDate.get(match.date).push(match);
  }

  for (const [date, group] of testByDate) {
    const priorMatches = matches.filter(
      match => match.date < date
    );

    evaluateDateGroup(
      group,
      priorMatches,
      ratings,
      metrics
    );
  }

  return {
    totalMatches: matches.length,
    trainingMatches: trainingMatches.length,
    testedMatches: testMatches.length,
    cutoffDate,
    metrics
  };
}

function printCalibration(modelName, metrics) {
  console.log(`\n${modelName} — Probability calibration`);
  console.log(
    "Outcome | Predicted range | Count | Avg predicted | Actual frequency"
  );

  for (const outcome of OUTCOMES) {
    const observations = metrics.calibration[outcome];

    for (let bin = 0; bin < CONFIG.calibrationBins; bin++) {
      const lower = bin / CONFIG.calibrationBins;
      const upper = (bin + 1) / CONFIG.calibrationBins;

      const selected = observations.filter(item => {
        if (bin === CONFIG.calibrationBins - 1) {
          return item.predicted >= lower && item.predicted <= upper;
        }

        return item.predicted >= lower && item.predicted < upper;
      });

      if (selected.length === 0) continue;

      const averagePredicted =
        selected.reduce((sum, item) => sum + item.predicted, 0) /
        selected.length;

      const actualFrequency =
        selected.reduce((sum, item) => sum + item.observed, 0) /
        selected.length;

      console.log(
        `${outcome.padEnd(7)} | ` +
        `${(lower * 100).toFixed(0).padStart(2)}–${(upper * 100).toFixed(0)}%`.padEnd(16) +
        ` | ${String(selected.length).padStart(5)} | ` +
        `${(averagePredicted * 100).toFixed(2).padStart(12)}% | ` +
        `${(actualFrequency * 100).toFixed(2).padStart(16)}%`
      );
    }
  }
}

function main() {
  const inputFiles = process.argv.slice(2);

  if (inputFiles.length === 0) {
    console.log(`
GoalMind AI — Historical Model Backtesting

Usage:
  node tools/model-backtest.js season.csv

Combined chronological evaluation:
  node tools/model-backtest.js season1.csv season2.csv season3.csv season4.csv

Supported CSV headers:
  Date,HomeTeam,AwayTeam,FTHG,FTAG

Also supports JSON arrays with date, homeTeam, awayTeam,
homeGoals and awayGoals.

This tool does not connect to Appwrite or change live predictions.
`);
    return;
  }

  const matches = loadMatches(inputFiles);
  const report = runBacktest(matches);

  console.log("\n==========================================");
  console.log("       GOALMIND AI BACKTEST REPORT");
  console.log("==========================================");
  console.log(`Datasets:           ${inputFiles.length}`);
  console.log(`Historical matches: ${report.totalMatches}`);
  console.log(`Training matches:   ${report.trainingMatches}`);
  console.log(`Matches evaluated:  ${report.testedMatches}`);
  console.log(`Training cutoff:    ${report.cutoffDate}`);

  console.log("\nMODEL COMPARISON");
  console.table([
    formatMetrics("Poisson", report.metrics.Poisson),
    formatMetrics("Elo", report.metrics.Elo),
    formatMetrics("Hybrid", report.metrics.Hybrid)
  ]);

  const bestBrier = Object.entries(report.metrics)
    .sort((a, b) =>
      a[1].brierTotal / a[1].matches -
      b[1].brierTotal / b[1].matches
    )[0];

  const bestLogLoss = Object.entries(report.metrics)
    .sort((a, b) =>
      a[1].logLossTotal / a[1].matches -
      b[1].logLossTotal / b[1].matches
    )[0];

  console.log(
    `Lowest Brier score: ${bestBrier[0]} ` +
    "(lower is better)."
  );

  console.log(
    `Lowest log loss:    ${bestLogLoss[0]} ` +
    "(lower is better)."
  );

  for (const modelName of ["Poisson", "Elo", "Hybrid"]) {
    printCalibration(modelName, report.metrics[modelName]);
  }

  console.log(
    "\nIMPORTANT: Calibration tables are diagnostics, not proof of " +
    "future accuracy. Compare models on identical test matches, and " +
    "do not treat this backtest as a guarantee of future performance."
  );
}

try {
  main();
} catch (error) {
  fail(error.message);
}

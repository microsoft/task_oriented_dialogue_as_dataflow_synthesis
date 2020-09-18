var pug = require('gulp-pug')
var gulp = require('gulp')
var rename = require('gulp-rename')
var data = require('gulp-data')
var connect = require('gulp-connect')
var replace = require('gulp-replace')
var ghPages = require('gulp-gh-pages')
var bower = require('gulp-bower')
var image = require('gulp-image')
var stylus = require('gulp-stylus')
var minify = require('gulp-minify')
var path = require('path')
var fs = require('fs')
var cheerio = require('cheerio')

var build_dir = 'task_oriented_dialogue_as_dataflow_synthesis/' // good to have this be the same as the repo name for gh-pages purposes

var rankEntries = function (entries) {
  entries.sort(function(a, b) {
    return (b.accuracy) - (a.accuracy);
  })  // First sort by average F1 + EM
/*
  entries.sort(function (a, b) {
    var f1Diff = Math.sign(b.f1 - a.f1)
    var emDiff = Math.sign(b.em - a.em)
    return f1Diff + emDiff
  })
  */

  for (var i = 0; i < entries.length; i++) {
    var entry = entries[i]
    if (i === 0) {
      entry.rank = 1
    } else {
      var prevEntry = entries[i - 1]
      var rank = prevEntry.rank
      if (entry.accuracy < prevEntry.accuracy) rank++
      entry.rank = rank
    }
  }
  return entries
}

function assert (condition, message) {
  if (!condition) {
    throw message || 'Assertion failed'
  }
}

var parseCompEntries = function (comp_file) {
  var leaderboard = require(comp_file).leaderboard
  var entries = []

  for (var i = 0; i < leaderboard.length; i++) {
    try {
      var o_entry = leaderboard[i]
      var entry = {}
      entry.user = o_entry.submission.user_name
      var description = o_entry.submission.description.trim()
      var regex_match = description.match(/(.*) ?\((.*)\) ?\((.*)\)(.*)/);
      if (regex_match) {
        entry.model_name = regex_match[1].trim() + " (" + regex_match[2].trim() + ")";
        entry.institution = regex_match[3].trim();
        if (regex_match[4].lastIndexOf('http') !== -1) {
          entry.link = regex_match[4].trim()
        }
      } else {
        entry.model_name = description.substr(0, description.lastIndexOf('(')).trim()
        var firstPart = description.substr(description.lastIndexOf('(') + 1)
        entry.institution = firstPart.substr(0, firstPart.lastIndexOf(')'))
        if (description.lastIndexOf('http') !== -1) {
          entry.link = description.substr(description.lastIndexOf('http')).trim()
        }
      }
      entry.date = o_entry.submission.created
      entry.accuracy = parseFloat(o_entry.scores.accuracy)
      if (entry.user === 'haptik101') {
        console.log(entry)
      }
      if (!(entry.accuracy >= 0)) throw 'Score invalid'
      if (entry.accuracy < 0.3) throw 'Score too low'
      //if (entry.model_name === '') {
      //  entry.model_name = 'Unnamed submission by ' + entry.user
      //}
      // if (entry.em > 50 && entry.f1 > 60) {
      if (entry.model_name !== '') {
        entries.push(entry);
      }
    } catch (err) {
      console.error(err)
      console.error(entry)
    }
  }
  entries = rankEntries(entries)
  return entries
}

var parseEntries = function (htmlStr) {
  var $ = cheerio.load(htmlStr)
  var parent = $('h1#leaderboard').closest('.ws-item').next()
  var entries = []
  $(parent).find('tbody > tr').each(function () {
    var entry = {}
    var cells = $(this).find('td')
    entry.description = cells.eq(1).text().trim()
    entry.model_name = entry.description.substr(0, entry.description.lastIndexOf('(')).trim()
    var firstPart = entry.description.substr(entry.description.lastIndexOf('(') + 1)
    entry.institution = firstPart.substr(0, firstPart.lastIndexOf(')'))
    var httpPos = entry.description.lastIndexOf('http')
    if (httpPos !== -1) {
      entry.link = entry.description.substr(entry.description.lastIndexOf('http')).trim()
    }
    delete entry.description
    entry.accuracy = parseFloat(cells.eq(3).text())
    entry.date = cells.eq(2).text().trim()
    entries.push(entry)
  })
  entries = rankEntries(entries)
  return entries
}

gulp.task('bower', function () {
  return bower()
    .pipe(gulp.dest('./' + build_dir + 'bower_components/'))
})

gulp.task('image', function () {
  return gulp.src('./views/images/*')
    .pipe(image())
    .pipe(gulp.dest('./' + build_dir))
})

gulp.task('js', function () {
  return gulp.src('./views/js/*')
    .pipe(minify())
    .pipe(gulp.dest('./' + build_dir + 'javascripts/'))
})

gulp.task('scrape_website', function (cb) {
  var Nightmare = require('nightmare')
  var fs = require('fs')
  var parse
  var nightmare = new Nightmare({
    switches: {
      'ignore-certificate-errors': true
    }
  })
  nightmare.goto('https://worksheets.codalab.org/worksheets/0xfc168b6317924c8a85df9ecd39bc52b2/')
  .wait(2000)
  .evaluate(function () {
    return document.body.innerHTML
  })
  .end()
  .then(function (result) {
    var jsonfile = require('jsonfile')
    var after = parseEntries(result)
    jsonfile.writeFile('./test.json', after, cb)
  })
})

gulp.task('connect', function () {
  connect.server({
    host: '0.0.0.0',
    root: '.'
  })
})

gulp.task('process_comp_output', function (cb) {
  var jsonfile = require('jsonfile')
  var entries2 = parseCompEntries('./out-v2.0.json')
  jsonfile.writeFile('./results2.0.json', entries2, cb)
})

gulp.task('generate_index', ['process_comp_output'], function () {
  var test_2 = require('./results2.0.json')
  var moment = require('moment')
  return gulp.src('views/index.pug')
      .pipe(data(function () {
        return {
          'test2': test_2,
          'moment': moment}
      }))
    .pipe(pug())
    .pipe(gulp.dest('./' + build_dir))
})

gulp.task('correct_link_paths', ['generate'], function () {
  return gulp.src('./' + build_dir + '**/*.html')
    .pipe(replace(/([^-](?:href|src)=[\'\"]\/)([^\'\"]*)([\'\"])/g, '$1' + build_dir + '$2$3'))
    .pipe(gulp.dest('./' + build_dir))
})

gulp.task('css', function () {
  return gulp.src('./views/styles/*.styl')
    .pipe(stylus())
    .pipe(gulp.dest('./' + build_dir + 'stylesheets'))
})

gulp.task('deploy', function () {
  return gulp.src('./' + build_dir + '**/*')
    .pipe(ghPages())
})

gulp.task('generate', ['bower', 'generate_index', 'process_comp_output'])
gulp.task('default', ['generate', 'correct_link_paths', 'image', 'js', 'css'])

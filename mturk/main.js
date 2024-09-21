$(document).ready(main);

const metadata = $("#metadata").attr("src"); // The location of the data file
let question_type; // Whether this is a "query" or a "validate" question

const paraphrase_prompt = ("<strong>Are all these questions asking the same thing?</strong>" +
                           " (It is okay if they are negations of each other.)");
const paraphrase_options = ["yes", "no"];


const controversial_prompt = ("<strong>How controversial are these questions?</strong>" +
                              " (How much would you expect random people to disagree about each?)");

const controversial_options = ["Very controversial",  "Somewhat controversial",
                               "Not very controversial", "Not at all controversial"];

if (metadata.startsWith("data:application/json;base64")) {
    const variables = JSON.parse(atob(metadata.substring(29)));
    question_type = variables["question_type"];
} else {
    d3.json(metadata, function(variables) {
        question_type = variables["question_type"];
    });
}

// From: https://developer.mozilla.org/en-US/docs/Glossary/Base64
function base64ToBytes(base64) {
    const binString = atob(base64);
    return Uint8Array.from(binString, (m) => m.codePointAt(0));
}

// From: https://developer.mozilla.org/en-US/docs/Glossary/Base64
function bytesToBase64(bytes) {
    const binString = Array.from(bytes, (x) => String.fromCodePoint(x)).join("");
    return btoa(binString);
}

function get_data(selector) {
    const id = $(selector).attr("id");
    const str = $("#" + id).val();
    return bytesToJson(str);
}

function bytesToJson(bytes) {
    const text = new TextDecoder().decode(base64ToBytes(bytes));
    return JSON.parse(text);
}

function main() {
    const data = get_data("#questions-values");
    for (let i = 0; i < data.length; i++) {
        const container = generateQuestionDiv(data[i], i);

        // Append the main question div to the questions container
        $("#questions-container").append(container);
    }
}

 function generateQuestionDiv(question_data, q_num) {
    // Create the main question div
    const questionDiv = $("<div></div>").addClass("question");

    // Create the question choice container div
    const questionChoiceContainer = $("<div></div>")
        .addClass("row question-choice-container col-12");

    // Create the inner div with the question text
    const questionTextDiv = $("<div></div>")
        .addClass("row col-12 border pt-1 pb-1 mb-1 ml-2 border-warning")
        .css("background-color", "#eee")
        .append($("<span></span>").text("Question " + (q_num + 1)));

    // Append the question text div to the question choice container
    questionChoiceContainer.append(questionTextDiv);

    // Create the row div for the question data
    const questionDataRow = $("<div></div>").addClass("row");

    // Create the question data div
    const questionDataDiv = $("<div></div>")
        .attr("id", "question-" + q_num)
        .addClass("question-data offset-1 col-11 pt-4");

    const question_conainer = make_question(question_data, q_num);

    questionDataDiv.append(question_conainer)

    // Append the question data div to the row div
    questionDataRow.append(questionDataDiv);

    // Append the question choice container and question data row to the main question div
    questionDiv.append(questionChoiceContainer);
    questionDiv.append(questionDataRow);
    return questionDiv;
}

function make_question(question_data, q_num) {
    let q_input;
    let attention_quesiton;
    if (question_type == "query") {
        const options = Object.keys(question_data["options"]);
        const question = question_data["question"];
        attention_quesiton = question_data["question"];

        q_input = make_query_question_input(question, options, q_num);
    } else if (question_type == "validate") {
        const questions = question_data["questions"];
        attention_quesiton = question_data["original"];
        q_input = make_validate_question_input(questions, attention_quesiton, q_num,
                                               paraphrase_options, paraphrase_prompt);

    } else {
        const questions = question_data["questions"];
        q_input = make_validate_question_input(questions, attention_quesiton, q_num,
                                               controversial_options, controversial_prompt);
    }

    const html_questions = $("<div>")
                             .attr("class", "col col-12")
                             .append(q_input);

    if (question_type == "query") {
        html_questions.append($("<hr />"))
                      .append(make_question_attention_check(attention_quesiton, q_num));
    }

    return html_questions;

}

function addOrdinalSuffix(number) {
    // Convert the number to a string
    const numStr = number.toString();

    // Get the last digit of the number
    const lastDigit = number % 10;

    // Get the last two digits of the number
    const lastTwoDigits = number % 100;

    // Determine the appropriate suffix
    let suffix = "th";
    if (lastTwoDigits < 11 || lastTwoDigits > 13) {
        if (lastDigit === 1) {
            suffix = "st";
        } else if (lastDigit === 2) {
            suffix = "nd";
        } else if (lastDigit === 3) {
            suffix = "rd";
        }
    }

    // Return the number with the suffix
    return numStr + suffix;
}

function make_question_attention_check(question, q_num) {
    const words = question.split(" ");
    const randomIndex = Math.floor(Math.random() * words.length);
    const answer = words[randomIndex];

    // Create the new question
    let question_reference;
    if (question_type == "query") {
        question_reference = " word of the above question?";
    } else if (question_type == "validate") {
        question_reference = " word of the the question, \"" + question + "\"";
    }
    const newQuestion = "What is the " + addOrdinalSuffix(randomIndex + 1) + question_reference;
    var outside_div = $("<div>")
                        .attr("class", "row")
                        .append($("<div>")
                                 .attr("class", "col col-md-6 col-sm-12")
                                 .append($("<p>").text(newQuestion))
                                )
                        .append($("<input>")
                                 .attr("name", "q_" + q_num + "_attn_answer")
                                 .attr("id", "q_" + q_num + "_attn" + "_answer")
                                 .attr("value", answer)
                                 .attr("hidden", true));

    var answer_div = $("<div>")
                    .attr("class", "mt-2 col-md-6 col-sm-12");

    for (let i = 0; i < words.length; i++) {
        answer_div.append($("<div>")
            .attr("class", "col-12")
            .append($("<input>")
                    .attr("type", "radio")
                    .attr("name", "q_" + q_num + "_attn")
                    .attr("id", "q_" + q_num + "_attn" + "_opt_" + i)
                    .attr("value", words[i])
                    .prop("required", true)
                 )
            .append($("<label>")
                    .text(words[i])
                 )
            );
    }

    outside_div.append(answer_div);

    return outside_div;
}

function make_validate_question_input(questions, original, q_num, options, prompt_text) {
    const prompt = $("<p>").html(prompt_text);
            
    const questions_list = $("<ul>");

    questions.forEach(function(question) {
        questions_list.append($("<li>").text(question));
    });

    var outside_div = $("<div>")
                        .attr("class", "row")
                        .append($("<div>")
                                 .attr("class", "col col-md-6 col-sm-12")
                                 .append(prompt)
                                 .append(questions_list)
                                 )
                        .append($("<input>")
                                 .attr("name", "q_" + q_num + "_question")
                                 .attr("id", "q_" + q_num + "_question")
                                 .attr("value", original)
                                 .attr("hidden", true));

    var prop_div = $("<div>")
                    .attr("class", "mt-2 col-md-6 col-sm-12");

    options.forEach(function(option) {
        prop_div.append($("<div>")
                          .attr("class", "col-12")
                          .append($("<input>")
                                    .attr("type", "radio")
                                    .attr("name", "q_" + q_num)
                                    .attr("id", "q_" + q_num + "_option_" + option)
                                    .attr("value", option)
                                    .prop("required",true)
                                 )
                          .append($("<label>")
                                    .text(option)
                                 )
                          );
    });

    outside_div.append(prop_div);

    return outside_div;
}

function make_query_question_input(question, options, q_num) {

    var outside_div = $("<div>")
                        .attr("class", "row")
                        .append($("<div>")
                                 .attr("class", "col col-md-6 col-sm-12")
                                 .append($("<p>")
                                        .html("<em>Please tell us your honest opinion on the folloiwng question.</em>"))
                                 .append($("<p>")
                                        .html("<strong>" + question + "</strong>"))
                                )
                        .append($("<input>")
                                 .attr("name", "q_" + q_num + "_question")
                                 .attr("id", "q_" + q_num + "_question")
                                 .attr("value", question)
                                 .attr("hidden", true))
                        .append($("<input>")
                                 .attr("name", "q_" + q_num + "_options")
                                 .attr("id", "q_" + q_num + "_options")
                                 .attr("value", options)
                                 .attr("hidden", true));

    var prop_div = $("<div>")
                    .attr("class", "mt-2 col-md-6 col-sm-12");

    const shuffled_options = shuffle(options);
    shuffled_options.forEach(function(option) {
        prop_div.append($("<div>")
                          .attr("class", "col-12")
                          .append($("<input>")
                                    .attr("type", "radio")
                                    .attr("name", "q_" + q_num)
                                    .attr("id", "q_" + q_num + "_option_" + option)
                                    .attr("value", option)
                                    .prop("required",true)
                                 )
                          .append($("<label>")
                                    .text(option)
                                 )
                          );
    });

    outside_div.append(prop_div);

    return outside_div;
}

// from : https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-arrays
function shuffle(unshuffled) {
    let shuffled = unshuffled
        .map(value => ({ value, sort: Math.random() }))
        .sort((a, b) => a.sort - b.sort)
        .map(({ value }) => value);
    return shuffled;
}

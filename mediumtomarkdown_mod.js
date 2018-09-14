const mediumToMarkdown = require('medium-to-markdown');
 
mediumToMarkdown.convertFromUrl("https://towardsdatascience.com/a-data-science-for-good-machine-learning-project-walk-through-in-python-part-two-2773bd52daf0").then(function (markdown) {
  console.log(markdown); //=> Markdown content of medium post
});

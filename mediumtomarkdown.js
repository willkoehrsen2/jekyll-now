const mediumToMarkdown = require('medium-to-markdown');
 
mediumToMarkdown.convertFromUrl('https://towardsdatascience.com/how-to-put-fully-interactive-runnable-code-in-a-medium-post-3dce6bdf4895')
.then(function (markdown) {
  console.log(markdown); //=> Markdown content of medium post
});

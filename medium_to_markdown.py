import copy
import subprocess

if __name__ == "__main__":
    post_url = input('Enter post url:')
    title = '-'.join(post_url.split('/')[-1].split('-')[:-1])
    date = input('Enter date (as 2018-10-05):')

    with open('mediumtomarkdown.js', 'r') as f:
        template = f.readlines()

    template_mod = copy.deepcopy(template)

    template_mod[2] = f'mediumToMarkdown.convertFromUrl("{post_url}")'

    with open('mediumtomarkdown_mod.js', 'w') as f:
        f.writelines(template_mod)

    post_dir = f'_posts/{date}-{title}.md'

    r = subprocess.call([f'node mediumtomarkdown_mod.js >> {post_dir}'],
                        shell = True)
    if r == 0:
        print(f'Post saved as markdown to {post_dir}')
    else:
        print('Error somewhere along the way.')

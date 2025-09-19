set nocompatible
syntax on
set termguicolors
filetype plugin indent on


call plug#begin('F:/Vim/neovim/nvim/plugged')
Plug 'joshdick/onedark.vim'
Plug 'neoclide/coc.nvim', {'branch': 'release'}
Plug 'morhetz/gruvbox'
Plug 'jiangmiao/auto-pairs'
call plug#end()

colorscheme gruvbox

hi Normal guibg=NONE ctermbg=NONE
hi NormalNC guibg=NONE ctermbg=NONE

set backspace=indent,eol,start

inoremap <expr> <TAB> pumvisible() ? "\<C-n>" : "\<Space>\<Space>\<Space>\<Space>"
inoremap <expr> <S-TAB> pumvisible() ? "\<C-p>" : "\<BS>\<BS>\<BS>\<BS>"

inoremap <silent><expr> <CR> pumvisible() ? coc#_select_confirm() : "\<CR>"

nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gr <Plug>(coc-references)

nnoremap <silent> K :call CocActionAsync('doHover')<CR>

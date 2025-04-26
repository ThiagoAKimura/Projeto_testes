# Análise de sentimentos

Se trata de um projeto aonde simulamos parte do CPA (avaliação semestral dos professores e cursos), mais especificamente os feedbacks escritos.
Há uma interface aonde o usuário poderá escrever um feedback sobre um tema de sua escolha, não necessariamente relacionado ao meio acadêmico, ou apenas expressar alguma ideia ou desabafo.
Depois será possivel clicar no botão de análise e será mostrado um pop-up com o resultado, deixando o a opção de corrigir a análise caso ache necessário, ou confirmar o resultado.
Há também um botão para visualizar um gráfico de pizza mostrando a proporção entre os tipos de resultados armazenados no banco, e outro botão para exportá-los.

## Tecnologias

- Python
- JavaScript
- HTML

## Instalação

Além da instalação do projeto, é necessário instalar essas bibliotecas para o backend:  
pip install flask flask-cors torch transformers deep-translator sqlalchemy openpyxl pandas matplotlib

# Acesse a pasta
No terminal, acesse a pasta cd /backend-main e execute o comando python ./app.py.  
Depois, nas pasta /frontend-main, pelo vscode, acesse a pasta /analise e execute o arquivo analise.html pelo live server, se certifique de que o server inicializou (app.py) antes de começar a testar.


var sentimentoResposta;

function limparTexto(){
    document.getElementById("campotext").value = "";
}

function abrirPopup() {
    document.getElementById('popup').style.display = 'flex';
}

  // Função para fechar o popup
function fecharPopup(event) {
    document.getElementById('popup').style.display = 'none';
    salvarFeedback(event);
}

function abrirAlertNoText() {
    const alerta = document.querySelector('.alert');
    alerta.style.display = 'flex';
    alerta.classList.remove('hide');
    setTimeout(() => {
        alerta.classList.add('hide');

        setTimeout(() => {
          alerta.style.display = 'none';
          alerta.classList.remove('hide');   // Limpa para próxima vez
        }, 500);
      }, 7000);
}

  // Função para fechar o popup
function fecharAlertNoText() {
    const alerta = document.querySelector('.alert');
    alerta.classList.add('hide');
  
    // Espera o tempo da transição para depois dar display: none
    setTimeout(() => {
      alerta.style.display = 'none';
      alerta.classList.remove('hide'); // opcional, caso queira mostrar de novo depois
    }, 500); 
}

function abrirAlertNotEnough() {
    const alerta = document.querySelector('.alert_notEnough');
    alerta.style.display = 'flex';
    alerta.classList.remove('hide');
    setTimeout(() => {
        alerta.classList.add('hide');

        setTimeout(() => {
          alerta.style.display = 'none';
          alerta.classList.remove('hide');   // Limpa para próxima vez
        }, 500);
      }, 7000);
}

  // Função para fechar o popup
function fecharAlertNotEnough() {
    const alerta = document.querySelector('.alert_notEnough');
    alerta.classList.add('hide');
  
    // Espera o tempo da transição para depois dar display: none
    setTimeout(() => {
      alerta.style.display = 'none';
      alerta.classList.remove('hide'); // opcional, caso queira mostrar de novo depois
    }, 500); 
}

function abrirAlertSavedText() {
    const alerta = document.querySelector('.alert_savedText');
    alerta.style.display = 'flex';
    alerta.classList.remove('hide');
    setTimeout(() => {
        alerta.classList.add('hide');

        setTimeout(() => {
          alerta.style.display = 'none';
          alerta.classList.remove('hide');   // Limpa para próxima vez
        }, 500);
      }, 7000);
}

  // Função para fechar o popup
function fecharAlertSavedText() {
    const alerta = document.querySelector('.alert_savedText');
    alerta.classList.add('hide');
  
    // Espera o tempo da transição para depois dar display: none
    setTimeout(() => {
      alerta.style.display = 'none';
      alerta.classList.remove('hide'); // opcional, caso queira mostrar de novo depois
    }, 500); 
}

function carregarGraficoSentimentos() {
    fetch('http://127.0.0.1:5000/grafico')
        .then(async res => {
            if (!res.ok) {
                const text = await res.text();
                throw new Error(`Erro ${res.status}: ${text}`);
            }
            return res.json();
        })
        .then(data => {
            if (data.image_base64) {
                document.getElementById('grafico').src = `data:image/png;base64,${data.image_base64}`;
            } else {
                console.warn("Imagem não recebida do backend.");
            }
        })
        .catch(err => {
            console.error("Erro ao carregar gráfico:", err.message);
        });

    const imagem = document.querySelector('.imagem_spam');
    imagem.style.display = 'flex';
}


document.addEventListener('DOMContentLoaded', function () {
    const imagemSpam = document.querySelector('.imagem_spam');

    if (imagemSpam) {
        imagemSpam.addEventListener('click', function (event) {
            // Verifica se o clique foi fora da imagem
            if (event.target === imagemSpam) {
                imagemSpam.style.display = 'none'; // Fecha a imagem
            }
        });
    }
});

function exportarParaExcel() {
    fetch("http://127.0.0.1:5000/exportar_excel", {
        method: "GET",
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Erro ao exportar dados');
        }
        return response.blob(); // Recebe o arquivo
    })
    .then(blob => {
        // Cria um link temporário para baixar o arquivo
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "dados_sentimentos.xlsx";  // Nome do arquivo para download
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    })
    .catch(error => {
        console.error("Erro ao tentar exportar o Excel:", error);
    });
}

async function salvarFeedback(event) {
    event.preventDefault();
    const texto = document.getElementById("campotext").value;
    sentimentoResposta = sentimentoResposta.toLowerCase();
    limparTexto();

    console.log("texto: ", texto);
    console.log("sentimento: ", sentimentoResposta);

    if (!texto || !sentimentoResposta) {
        alert("Texto ou sentimento ausente.");
        return;
    }

    try {
        console.log("entrou no try")
        const response = await fetch("http://127.0.0.1:5000/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: texto, sentiment: sentimentoResposta })
        });

        const data = await response.json();

        if (data.error) {
            alert("Erro ao salvar: " + data.error);
        } else {
            console.log("Feedback salvo com sucesso.");
        }

    } catch (error) {
        console.error("Erro ao salvar feedback:", error);
    }
}

function clickYes(){
    document.getElementById("pergunta-popup").innerHTML = `Ótimo! Muito obrigado por sua participação.`;

    document.getElementById("div-sim-nao").style.display = `none`;
    document.getElementById("div-options").style.display = `none`;
    document.getElementById("div-success").style.display = `flex`;
}

function clickNo(){
    document.getElementById("pergunta-popup").innerHTML = `Certo, então qual destas opções melhor classifica seu feedback?`;

    document.getElementById("div-sim-nao").style.display = `none`;
    document.getElementById("div-options").style.display = `flex`;
    document.getElementById("div-success").style.display = `none`;
}

function clickPos(){
    if(sentimentoResposta == 'POSITIVO')
        document.getElementById("pergunta-popup").innerHTML = `Entendi, talvez tenhamos acertado. Muito obrigado pela participação!`;
    else{
        document.getElementById("pergunta-popup").innerHTML = `Anotado, muito obrigado pela ajuda e pela participação!`;
    }

    document.getElementById("div-sim-nao").style.display = `none`;
    document.getElementById("div-options").style.display = `none`;
    document.getElementById("div-success").style.display = `flex`;

    sentimentoResposta = 'positivo'
}

function clickNeg(){
    if(sentimentoResposta == 'NEGATIVO')
        document.getElementById("pergunta-popup").innerHTML = `Entendi, talvez tenhamos acertado. Muito obrigado pela participação!`;
    else{
        document.getElementById("pergunta-popup").innerHTML = `Anotado, muito obrigado pela ajuda e pela participação!`;
    }

    document.getElementById("div-sim-nao").style.display = `none`;
    document.getElementById("div-options").style.display = `none`;
    document.getElementById("div-success").style.display = `flex`;

    sentimentoResposta = 'negativo'
}

function clickNtl(){
    if(sentimentoResposta == 'NEUTRO')
        document.getElementById("pergunta-popup").innerHTML = `Entendi, talvez tenhamos acertado. Muito obrigado pela participação!`;
    else{
        document.getElementById("pergunta-popup").innerHTML = `Anotado, muito obrigado pela ajuda e pela participação!`;
    }

    document.getElementById("div-sim-nao").style.display = `none`;
    document.getElementById("div-options").style.display = `none`;
    document.getElementById("div-success").style.display = `flex`;

    sentimentoResposta = 'neutro'
}

async function analise(){
    let campoTexto = document.getElementById("campotext");
    let texto = campoTexto.value;

    if(texto.length < 1) {
        abrirAlertNoText();
        return;
    }

    if(texto.length < 4) {
        abrirAlertNotEnough();
        return;
    }

    // Mostra carregamento
    document.getElementById("loading").style.display = 'flex';

    try {
        const response = await fetch('http://127.0.0.1:5000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: texto })
        });

        const data = await response.json();

        if (data.error) {
            alert("Erro: " + data.error);
            return;
        }

        var sentimento;
        var image;
        var color;

        switch(data.sentiment){
            case 'positive':
                sentimento = 'POSITIVO';
                image = '../assets/positive-vote.png';
                color = '#3b7d3c'
                break;
            case 'negative':
                sentimento = 'NEGATIVO';
                image = '../assets/negative-vote.png';
                color = '#a61d16'
                break;
            case 'neutral':
                sentimento = 'NEUTRO';
                image = '../assets/line.png';
                color = '#807c7c'
                break;
            default:
                break;
        }

        sentimentoResposta = '';
        sentimentoResposta = sentimento
        

        document.getElementById("div-sim-nao").style.display = `flex`;
        document.getElementById("div-options").style.display = `none`;
        document.getElementById("div-success").style.display = `none`;

        document.getElementById("pergunta-popup").innerHTML = `A análise do seu texto está correta?`;
        document.getElementById("responseText").innerHTML = `<strong>${sentimento}</strong>`;
        document.getElementById("responseText").style.color = `${color}`;
        document.getElementById("header-popup").style["background-color"] = `${color}`;
        document.getElementById("imageFeedback").src = `${image}`
        abrirPopup();

    } catch (error) {
        alert("Erro ao conectar ao servidor.");
        console.error(error);
    } finally {
        // Esconde carregamento
        document.getElementById("loading").style.display = 'none';
    }
}


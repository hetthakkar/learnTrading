import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from sklearn.metrics import confusion_matrix
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.graph_objects as go
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from feature_augment import *
from sklearn.svm import SVC # "Support vector classifier"
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, accuracy_score 

app = dash.Dash(external_stylesheets=[dbc.themes.PULSE])


df = pd.read_csv('NIFTYOHLC.csv')
df.set_index("Date",inplace = True)

df['Target'] = df['Close'].shift(-1)-df['Close'] 
df['Target'] = df['Target']/abs(df['Target'])
df.dropna(inplace = True)
df.tail()



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['Open','High', 'Low', 'Close']], df['Target'], test_size=0.25,shuffle = False)

print("Train data {0}".format(X_train.shape))
print("Test data {0}".format(X_test.shape))

feature_X_train = create_features(X_train)
feature_X_test = create_features(X_test)

features = feature_X_train.columns

feature_checklist_options = []

for feature in features:
    feature_checklist_options.append({"label" : feature, "value" : feature})



def get_confusion_matrix_fig(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    mat[[0,1]] = mat[[1,0]]
    fig = go.Figure(
            data=go.Heatmap(
                            z=mat,
                            colorscale = 'Viridis', 
                            x = ['Up', 'Down'],
                            y = ['Down', 'Up'],
        ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        
    )
    return fig

additional_params = [
    [ #params for polynomial kernel
        dbc.Row(html.P("gamma : ", style = {"padding-top" : "1em", "padding-bottom" : "1em"})),
        dbc.Row(html.Div(dcc.Slider( #gamma param
            id = {"type" : "svm_param", "name": "gamma"},
            min = -5,
            max = 0,
            marks={i: str((10**i)) for i in range(-5,1)},
            step = 1,
            value = -2
        ), style = {"width" : "20em"})),

        dbc.Row(html.P("degree : ", style = {"padding-top" : "1em", "padding-bottom" : "1em"})),
        dbc.Row(html.Div(dcc.Slider( #degree param
            id = {"type" : "svm_param", "name": "degree"},
            min = 0,
            max = 5,
            marks={i: str(i) for i in range(6)},
            value = 1
        ), style = {"width" : "20em"})),

        dbc.Row(html.P("coef0 : ", style = {"padding-top" : "1em", "padding-bottom" : "1em"})),
        dbc.Row(html.Div(dcc.Slider( #coef0 param
            id = {"type" : "svm_param", "name": "coef0"},
            min = 0,
            max = 10,
            marks={i: str(i) for i in range(11)},
            value = 1
        ), style = {"width" : "20em"})),
    ],
    [ #params for rbf kernel
        dbc.Row(html.P("gamma : ", style = {"padding-top" : "1em", "padding-bottom" : "1em"})),
        dbc.Row(html.Div(dcc.Slider( #gamma param
            id = {"type" : "svm_param", "name": "gamma"},
            min = -5,
            max = 0,
            marks={i: str((10**i)) for i in range(-5,1)},
            step = 1,
            value = -2
        ), style = {"width" : "20em"})),
    ],
    [
        dbc.Row(html.P("coef0 : ", style = {"padding-top" : "1em", "padding-bottom" : "1em"})),
        dbc.Row(html.Div(dcc.Slider( #coef0 param
            id = {"type" : "svm_param", "name": "coef0"},
            min = 0,
            max = 10,
            marks={i: str(i) for i in range(11)},
            value = 1
        ), style = {"width" : "20em"})),
    ]
]

parameters_svm = [
    dbc.Row(html.P("Parameters : ")),
    dbc.Row(
        
        html.Div(
            dcc.Dropdown(
                id = {"type" : "svm_param", "name" : "kernel"},
                placeholder = "Select Kernel",
                options = [
                    {'label' : 'rbf', 'value' : 'rbf'},
                    {'label' : 'polynomial', 'value' : 'poly'},
                    {'label' : 'sigmoid', 'value' : 'sigmoid'}
            ]),
            style = {"width" : "15em"}
        )
        
    ),

    dbc.Row(html.P("C : ", style = {"padding-top" : "1em", "padding-bottom" : "1em"})),
    dbc.Row(
        html.Div(
            dcc.Slider( #C param
                id = {'type' : "svm_param", 'name' : "C"},
                min = -2,
                max = 4,
                marks={i: str(10**i) for i in range(-2,5)},
                step = 1,
                value = 0,
            ),
            style = {"width" : "20em"}
        )
        
    ),
    dbc.Row(html.Div(
        id = "svm_additional",
        children = [],
        style = {"padding-left" : "1em"}
    ))
]

app.layout = html.Div([
    dbc.Row(
        dbc.Col(
            dbc.Jumbotron(
                dbc.Container([
                    html.H1("QuantML", className="text-center"),
                    html.P("Visualize and Test Machine Learning models on Financial Data", className="lead text-center")
                ]), 
                 
                fluid = True,
                style = {
                    "padding" : "2em"
                }
            ),
           
        )
    ),
    dbc.Row([
        dbc.Col(
            children = [
                
                dbc.Row(
                    className = "pp",
                    children = [
                        html.Div(
                        dcc.Dropdown(
                            id = "model_select", 
                            placeholder = "Select Model",
                            options = [
                                {'label' : 'SVM', 'value' : 'svm'},
                                {'label' : 'Untitled', 'value' : 'unt'}
                            ],
                            value = []
                        ), style = {"width" : "20em", "padding-bottom" : "2em", "padding-left" : "2em"})
                    ],
                    align = "start"
                ),
                dbc.Row(
                    html.Div(id = "model_parameters", children = [], style = {"padding" : "1em", "padding-left" : "3em"}),
                    align = "start"
                    
                )
                
                # html.Div(
                #     className = "d-flex justify-content-center",
                #     children = parameters_svm
                # )

            ],
            
        
        ),
        dbc.Col(
            children = [
                dbc.Row(
                    html.H5("Select Features: ")
                ),
                dbc.Row(
                    dbc.Checklist(
                        id = "feature_checklist",
                        options = feature_checklist_options,
                        value = ['rsi14', '20DMA'],
                        style = {"padding-left" : "2em", "padding-top" : "1em"}
                    )
                )
            ]
        ),
        dbc.Col(
            
            children = [
                dbc.Row(html.H4("Performance", style = {"padding-left" : "0em","padding-bottom" : "1em"})),

                dbc.Row([
                    dbc.Col(html.P("Accuracy: "), width = 2),
                    dbc.Col(html.P(id = "acc_score", children = "Insufficient Params"))
                ], justify = "start"),

                dbc.Row(
                    dcc.Graph(
                        id = "confusion_matrix",
                        figure = go.Figure()
                    )
                )

            ]
        ),   
            
    ],justify="center"),

],
    style = {
        # "margin-left" : "5px",
        "padding" : "5em",
        "padding-top" : "2em"
    }

)

@app.callback(
    Output('model_parameters', 'children'),
    [Input('model_select', 'value')]
)
def render_parameters(model_name):
    if(model_name == "svm"):
        return parameters_svm
    else:
        return []

@app.callback(
    Output('svm_additional', 'children'),
    [Input({"type" : "svm_param", "name" : "kernel"}, 'value')]
)
def render_parameters(kernel_name):
    if(kernel_name == "poly"):
        return additional_params[0]
    elif(kernel_name == "rbf"):
        return additional_params[1]
    elif(kernel_name == "sigmoid"):
        return additional_params[2]
    else:
        return []



@app.callback(
    [Output('acc_score', 'children'), Output('confusion_matrix', 'figure')],
    [Input("model_select", "value"),Input("feature_checklist", "value"), Input({"type" : "svm_param", "name" : ALL}, 'value')],
    [State({'type': 'svm_param', 'name': ALL}, 'id')]
)
def calc_accuracy(model_name, feature_filter, values, ids):

    fig = go.Figure()
    print("------------------")
    # print(values, ids)
    # print(values, ids)
    if(model_name == "svm"):
        svm_model_params = {}
        for value,id_ in zip(values, ids):
            if (value is None):
                # print(value, id_)
                return "Insufficient Params", fig
            svm_model_params[id_["name"]] = value

        # print("Done")
        

        if("C" in svm_model_params.keys()):
            svm_model_params["C"] = 10**svm_model_params["C"]

        if("gamma" in svm_model_params.keys()):
            svm_model_params["gamma"] = 10**svm_model_params["gamma"]

        # print(svm_model_params)
        mysvm = SVC(**svm_model_params)

        local_X_train = feature_X_train[feature_filter]
        local_X_test = feature_X_test[feature_filter]

        print("Fitting")
        mysvm.fit(local_X_train, y_train)
        print("Fitting done")
        y_pred = mysvm.predict(local_X_test)

        precision = precision_score(y_test, y_pred, average='binary')
        acc = accuracy_score(y_test,y_pred)
        # print("Accuracy is " ,acc)
        return str(acc), get_confusion_matrix_fig(y_test,y_pred)

    else:
        return "", fig
 
app.config['suppress_callback_exceptions'] = True
if __name__ == "__main__":
    app.run_server(debug = True)



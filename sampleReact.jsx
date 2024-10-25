function TestComponent() {
    // ruleid:react-dangerouslysetinnerhtml
    return <div dangerouslySetInnerHTML={createMarkup()} />;
}

function OkComponent() {
    // OK
    return {__html: 'Первый &middot; Второй'};
}


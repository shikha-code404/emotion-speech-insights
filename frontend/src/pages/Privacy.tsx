export default function Privacy() {
  return (
    <div className="space-y-6">
      <h1 className="h1">Privacy</h1>
      <section className="card prose-body">
        <h2 className="h2 mb-3">What happens to your voice clip</h2>
        <p>Nothing is recorded until you tick the consent box and press the record button. Your browser asks for microphone permission, and the microphone is switched off as soon as you stop or cancel.</p>
        <p>When you press "Analyse", the clip is sent to the server, converted, analysed, and then deleted straight away. The server keeps no copy, and the clip is not used for training.</p>
        <p>For playback on the results page, a copy stays in this browser tab's memory only. It is removed when you leave the results page or close the tab.</p>
      </section>
      <section className="card prose-body">
        <h2 className="h2 mb-3">What happens to your questionnaire answers</h2>
        <p>Your answers are sent to the server only to be scored. They are not stored, logged or linked to you.</p>
      </section>
      <section className="card prose-body">
        <h2 className="h2 mb-3">What this site does not do</h2>
        <p>There are no accounts, no history and no cookies or local storage for your results. Nothing is saved between visits.</p>
        <p>The voice result and the questionnaire result are never combined, and neither is a diagnosis.</p>
      </section>
    </div>
  );
}

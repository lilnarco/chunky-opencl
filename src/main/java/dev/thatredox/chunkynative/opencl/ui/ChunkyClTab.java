package dev.thatredox.chunkynative.opencl.ui;

import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.context.KernelLoader;
import dev.thatredox.chunkynative.opencl.renderer.OidnDenoiser;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.util.Duration;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.ui.render.RenderControlsTab;

import javafx.scene.control.Slider;
import javafx.scene.layout.HBox;

import se.llbit.chunky.PersistentSettings;

import java.io.File;

public class ChunkyClTab implements RenderControlsTab {
    protected final VBox box;
    private Scene scene;
    private final Label renderTimeLabel;
    private final Label buildLabel;
    private final Timeline renderTimeTicker;

    public static float russianRouletteThreshold = 50.0f;
    public static boolean profileRender = false;
    // Water shader override: null = scene default, 0 = still, 1 = simplex.
    public static Integer waterShaderOverrideId = null;

    public ChunkyClTab(Scene scene) {
        this.scene = scene;

        box = new VBox(10.0);
        box.setPadding(new Insets(10.0));

        renderTimeLabel = new Label();
        buildLabel = new Label();
        updateRenderTimeLabel();
        box.getChildren().add(renderTimeLabel);
        box.getChildren().add(buildLabel);

        // Russian Roulette UI
        Label rrLabel = new Label("Russian Roulette Threshold: 50%");
        Slider rrSlider = new Slider(0, 100, 50);
        rrSlider.setShowTickLabels(true);
        rrSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            russianRouletteThreshold = newVal.floatValue();
            rrLabel.setText(String.format("Russian Roulette Threshold: %d%%", (int)russianRouletteThreshold));
            scene.softRefresh();
        });
        box.getChildren().addAll(rrLabel, rrSlider);

        // Water shader selector: overrides the scene's water shader (Still/Simplex).
        Label waterShaderLabel = new Label("Water shader:");
        javafx.scene.control.ComboBox<String> waterShaderCombo = new javafx.scene.control.ComboBox<>();
        waterShaderCombo.getItems().addAll("Scene default", "Still", "Simplex");
        waterShaderCombo.getSelectionModel().select(
                waterShaderOverrideId == null ? 0 : waterShaderOverrideId + 1);
        waterShaderCombo.valueProperty().addListener((obs, oldVal, newVal) -> {
            int index = waterShaderCombo.getSelectionModel().getSelectedIndex();
            waterShaderOverrideId = index <= 0 ? null : index - 1;
            scene.softRefresh();
        });
        HBox waterShaderRow = new HBox(10.0, waterShaderLabel, waterShaderCombo);
        box.getChildren().add(waterShaderRow);

        // Kernel profiling UI
        CheckBox profileCheck = new CheckBox("Profile render (kernel operation counters)");
        profileCheck.setSelected(ChunkyClTab.profileRender);
        profileCheck.selectedProperty().addListener((obs, oldVal, newVal) -> {
            ChunkyClTab.profileRender = newVal;
            scene.softRefresh();
        });
        box.getChildren().add(profileCheck);

        // OIDN denoiser UI
        Label denoiseLabel = new Label("OIDN Denoiser:");
        CheckBox denoiseCheck = new CheckBox("Denoise final image with OIDN");
        denoiseCheck.setSelected(OidnDenoiser.enabled);
        denoiseCheck.selectedProperty().addListener((obs, oldVal, newVal) -> {
            OidnDenoiser.enabled = newVal;
        });

        Label binaryLabel = new Label("OIDN binary path:");
        OidnDenoiser.binaryPath = PersistentSettings.settings.getString("clOidnBinaryPath", OidnDenoiser.binaryPath);
        TextField binaryField = new TextField(OidnDenoiser.binaryPath);
        binaryField.setPromptText("oidnDenoise");
        binaryField.setOnAction(event -> saveBinaryPath(binaryField));
        binaryField.focusedProperty().addListener((obs, wasFocused, isFocused) -> {
            if (!isFocused) {
                saveBinaryPath(binaryField);
            }
        });
        Button browseButton = new Button("Browse...");
        browseButton.setOnMouseClicked(event -> {
            FileChooser chooser = new FileChooser();
            chooser.setTitle("Select OIDN binary");
            File file = chooser.showOpenDialog(null);
            if (file != null) {
                OidnDenoiser.binaryPath = file.getAbsolutePath();
                binaryField.setText(file.getAbsolutePath());
                saveBinaryPath(binaryField);
            }
        });
        HBox binaryRow = new HBox(10.0, browseButton, binaryField);
        HBox.setHgrow(binaryField, javafx.scene.layout.Priority.ALWAYS);

        Button denoiseNowButton = new Button("Denoise now");
        denoiseNowButton.setOnMouseClicked(event -> {
            OidnDenoiser.triggerDenoise = true;
            if (!OpenClRenderTimer.isRunning()) {
                // Restart the render so it can consume the trigger flag.
                scene.softRefresh();
            }
        });

        Label denoiseHint = new Label("Runs at the end of a completed render, or immediately via \"Denoise now\". The result is saved as <scene name>_denoised.png");
        denoiseHint.setWrapText(true);

        box.getChildren().addAll(denoiseLabel, denoiseCheck, binaryLabel, binaryRow,
                denoiseNowButton, denoiseHint);

        Button deviceSelectorButton = new Button("Select OpenCL Device");
        deviceSelectorButton.setOnMouseClicked(event -> {
            DeviceSelector selector = new DeviceSelector();
            selector.show();
        });
        box.getChildren().add(deviceSelectorButton);

        if (KernelLoader.canHotReload()) {
            Button reloadButton = new Button("Reload!");
            reloadButton.setOnMouseClicked(event -> {
                ContextManager.reload();
                scene.refresh();
            });
            box.getChildren().add(reloadButton);
        }

        renderTimeTicker = new Timeline(
                new KeyFrame(Duration.millis(200), event -> updateRenderTimeLabel())
        );
        renderTimeTicker.setCycleCount(Timeline.INDEFINITE);
        renderTimeTicker.play();
    }

    @Override
    public void update(Scene scene) {
        this.scene = scene;
        updateRenderTimeLabel();
    }

    @Override
    public String getTabTitle() {
        return "OpenCL";
    }

    @Override
    public Node getTabContent() {
        return box;
    }

    private void saveBinaryPath(TextField field) {
        OidnDenoiser.binaryPath = field.getText().trim();
        if (OidnDenoiser.binaryPath.isEmpty()) {
            OidnDenoiser.binaryPath = "oidnDenoise";
        }
        PersistentSettings.settings.setString("clOidnBinaryPath", OidnDenoiser.binaryPath);
        PersistentSettings.save();
    }

    private void updateRenderTimeLabel() {
        // A blocking kernel build (JIT specialization or first launch) shows here so
        // 0 SPP during a ~1 min compile doesn't look frozen. Set on the render
        // thread, read here on the FX ticker thread.
        String compiling = ContextManager.Renderer.compileStatus;
        if (compiling != null) {
            renderTimeLabel.setText(compiling);
            return;
        }
        long millis = OpenClRenderTimer.getElapsedMillis();
        double seconds = millis / 1000.0;
        double compileSeconds = OpenClRenderTimer.getCompileMillis() / 1000.0;
        String suffix = OpenClRenderTimer.isRunning() ? " (running)" : "";
        renderTimeLabel.setText(String.format("Render Time: %.1f s (compile %.1f s)%s",
                seconds, compileSeconds, suffix));
        // Null-guard: the constructor's first update call predates nothing now, but
        // the ticker can race tab teardown — never let the label die over a status.
        if (buildLabel != null) {
            String outcome = ContextManager.Renderer.lastBuildOutcome;
            buildLabel.setText(outcome != null ? outcome : "");
        }
    }
}
